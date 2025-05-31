provider "aws" {
  region = var.aws_region
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket" "backup_bucket" {
  bucket = "${var.backup_bucket_name_prefix}-${random_string.bucket_suffix.result}"
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "backup_bucket_versioning" {
  bucket = aws_s3_bucket.backup_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backup_bucket_lifecycle" {
  bucket = aws_s3_bucket.backup_bucket.id

  rule {
    id     = "expire-backups"
    status = "Enabled"

    expiration {
      days = var.s3_lifecycle_expiration_days
    }

    # It's also good practice to clean up incomplete multipart uploads
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_public_access_block" "backup_bucket_public_access" {
  bucket                  = aws_s3_bucket.backup_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Role for the ECS Backup Task
resource "aws_iam_role" "ecs_backup_task_role" {
  name               = "${var.backup_bucket_name_prefix}-ecs-task-role-${random_string.bucket_suffix.result}"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
  tags = var.tags
}

resource "aws_iam_policy" "ecs_backup_task_policy" {
  name        = "${var.backup_bucket_name_prefix}-ecs-task-policy-${random_string.bucket_suffix.result}"
  description = "Policy for ECS backup task to access S3 and CloudWatch Logs"
  policy      = jsonencode({
    Version   = "2012-10-17",
    Statement = [
      {
        Action   = [
          "s3:PutObject",
          "s3:GetObject", # Needed if the sync does checks
          "s3:ListBucket", # Needed for sync
          "s3:DeleteObject" # Needed if sync uses --delete
        ],
        Effect   = "Allow",
        Resource = [
          aws_s3_bucket.backup_bucket.arn,
          "${aws_s3_bucket.backup_bucket.arn}/*"
        ]
      },
      {
        Action   = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Effect   = "Allow",
        Resource = aws_cloudwatch_log_group.backup_log_group.arn
      }
    ]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "ecs_backup_task_role_attachment" {
  role       = aws_iam_role.ecs_backup_task_role.name
  policy_arn = aws_iam_policy.ecs_backup_task_policy.arn
}

# CloudWatch Log Group for the backup task
resource "aws_cloudwatch_log_group" "backup_log_group" {
  name              = var.cloudwatch_log_group_name
  retention_in_days = var.s3_lifecycle_expiration_days # Align log retention with backup retention
  tags              = var.tags
}

# ECS Task Definition for the Backup Job
# This task definition assumes that the volume specified by 'var.lancedb_data_volume_name'
# is available and defined in the ECS cluster, typically by the main LanceDB service/task.
# The backup task will mount this existing volume.
resource "aws_ecs_task_definition" "backup_task_def" {
  family                   = "${var.backup_bucket_name_prefix}-backup-task"
  network_mode             = "awsvpc" # Required for Fargate
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.backup_sidecar_cpu
  memory                   = var.backup_sidecar_memory
  execution_role_arn       = aws_iam_role.ecs_backup_task_role.arn # For pulling image and CW logs
  task_role_arn            = aws_iam_role.ecs_backup_task_role.arn # For S3 access from container

  # This volume definition refers to a volume that should be defined and managed
  # by the main LanceDB service's task definition. The backup task expects this volume
  # to exist and be mountable.
  volume {
    name = var.lancedb_data_volume_name
    # If using EFS, configure efs_volume_configuration here.
    # If it's a Docker volume managed by ECS (e.g., for ephemeral storage shared between sidecars),
    # this definition is okay. The actual data persistence relies on how LanceDB itself is configured.
    # For this module, we assume the volume exists and is populated by LanceDB.
  }

  container_definitions = jsonencode([
    {
      name      = "${var.backup_bucket_name_prefix}-sidecar"
      image     = var.backup_sidecar_image
      cpu       = var.backup_sidecar_cpu
      memory    = var.backup_sidecar_memory
      essential = true
      # Command to sync data to S3.
      # Ensure the image has AWS CLI and necessary tools.
      # The sync command will run, and then the container will exit.
      command   = [
        "sh", "-c",
        "aws s3 sync ${var.lancedb_data_path_in_container} s3://${aws_s3_bucket.backup_bucket.id}/ --delete --only-show-errors"
      ]
      mountPoints = [
        {
          sourceVolume  = var.lancedb_data_volume_name
          containerPath = var.lancedb_data_path_in_container
          readOnly      = true # Backup should not modify the source data
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.backup_log_group.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs-backup"
        }
      }
    }
  ])
  tags = var.tags
}

# CloudWatch Event Rule (Scheduler)
resource "aws_cloudwatch_event_rule" "backup_schedule_rule" {
  name                = "${var.backup_bucket_name_prefix}-schedule-rule-${random_string.bucket_suffix.result}"
  description         = "Schedules the LanceDB S3 backup task"
  schedule_expression = var.backup_schedule_expression
  tags                = var.tags
}

# IAM Role for CloudWatch Events to trigger ECS Task
resource "aws_iam_role" "events_invoke_ecs_role" {
  name = "${var.backup_bucket_name_prefix}-events-invoke-ecs-role-${random_string.bucket_suffix.result}"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Action    = "sts:AssumeRole",
      Effect    = "Allow",
      Principal = {
        Service = "events.amazonaws.com"
      }
    }]
  })
  tags = var.tags
}

resource "aws_iam_policy" "events_invoke_ecs_policy" {
  name        = "${var.backup_bucket_name_prefix}-events-invoke-ecs-policy-${random_string.bucket_suffix.result}"
  description = "Allows CloudWatch Events to run ECS tasks"
  policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{
      Effect   = "Allow",
      Action   = "ecs:RunTask",
      Resource = "*", # Should be restricted to the specific task definition if possible, but requires knowing the account and region.
                      # For a generic module, "*" is sometimes used, or requires more inputs.
                      # Let's try to scope it down.
      # Resource = "arn:aws:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:task-definition/${aws_ecs_task_definition.backup_task_def.family}:*"
      # Using "*" for now due to potential complexity with data source for account ID in module context or if family changes.
      # A better approach would be to construct the ARN precisely.
      Condition = {
        StringEquals = {
          "ecs:cluster" = var.ecs_cluster_arn
        }
      }
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "events_invoke_ecs_attachment" {
  role       = aws_iam_role.events_invoke_ecs_role.name
  policy_arn = aws_iam_policy.events_invoke_ecs_policy.arn
}

# CloudWatch Event Target - The ECS Task
resource "aws_cloudwatch_event_target" "backup_ecs_target" {
  rule      = aws_cloudwatch_event_rule.backup_schedule_rule.name
  arn       = var.ecs_cluster_arn # ARN of the ECS cluster
  role_arn  = aws_iam_role.events_invoke_ecs_role.arn
  target_id = "${var.backup_bucket_name_prefix}-backup-target"

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.backup_task_def.arn
    launch_type         = "FARGATE" # Ensure Fargate is specified
    platform_version    = "LATEST"  # Or a specific Fargate platform version

    network_configuration {
      # Subnets and security groups should be provided by the user of the module,
      # as they are specific to the VPC where the ECS cluster runs.
      # These need to allow outbound internet access for S3 and CloudWatch.
      subnets         = var.fargate_subnets # This variable needs to be added to variables.tf
      security_groups = var.fargate_security_groups # This variable needs to be added to variables.tf
    }
  }
}

# Data source for AWS Account ID (if needed for more specific IAM policies)
# data "aws_caller_identity" "current" {}
