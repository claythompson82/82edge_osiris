variable "aws_region" {
  description = "AWS region where resources will be created."
  type        = string
  default     = "us-east-1"
}

variable "tags" {
  description = "A map of tags to assign to created resources."
  type        = map(string)
  default     = {}
}

variable "backup_bucket_name_prefix" {
  description = "Prefix for the S3 backup bucket name. A random suffix will be appended."
  type        = string
  default     = "lancedb-backup"
}

variable "backup_schedule_expression" {
  description = "Cron expression for the backup schedule (e.g., 'cron(0 2 * * ? *)' for daily at 2 AM UTC)."
  type        = string
  default     = "cron(0 2 * * ? *)" # Default to daily at 2 AM UTC
}

variable "ecs_cluster_arn" {
  description = "ARN of the ECS cluster where the backup task will run."
  type        = string
  # No default, this must be provided by the calling module
}

variable "lancedb_service_name" {
  description = "Name of the LanceDB ECS service. Used to discover the running task for data volume access (alternative to direct volume mapping if not feasible)."
  type        = string
  # This might be needed if the backup task needs to find where LanceDB is running.
  # However, a simpler approach is to run the backup task on the same task definition or a sidecar.
  # For now, let's assume the backup task will have direct access to a volume.
}

variable "lancedb_task_definition_arn" {
  description = "ARN of the LanceDB ECS task definition. The backup sidecar will be added to this or a similar definition."
  type        = string
  # This is crucial for defining the backup task, assuming it shares a volume.
}

variable "lancedb_container_name" {
  description = "Name of the LanceDB container within the task definition."
  type        = string
  default     = "lancedb" # Matches the service name in docker-compose.cloud.yaml
}

variable "lancedb_data_volume_name" {
  description = "Name of the volume in the LanceDB task definition that stores /app/lancedb_data."
  type        = string
  default     = "lancedb_data_volume" # Matches the volume in docker-compose.cloud.yaml
}

variable "lancedb_data_path_in_container" {
  description = "Path inside the LanceDB container where data is stored."
  type        = string
  default     = "/app/lancedb_data"
}

variable "backup_sidecar_image" {
  description = "Docker image for the backup sidecar container (must include AWS CLI and a sync script)."
  type        = string
  default     = "amazon/aws-cli:latest" # A basic image with AWS CLI. A custom script execution will be needed.
}

variable "backup_sidecar_cpu" {
  description = "CPU units for the backup sidecar container."
  type        = number
  default     = 256
}

variable "backup_sidecar_memory" {
  description = "Memory in MiB for the backup sidecar container."
  type        = number
  default     = 512
}

variable "s3_lifecycle_expiration_days" {
  description = "Number of days after which to expire objects in the backup S3 bucket."
  type        = number
  default     = 30
}

variable "cloudwatch_log_group_name" {
  description = "CloudWatch Log Group name for the backup task logs."
  type        = string
  default     = "/ecs/lancedb-backup-task"
}

variable "fargate_subnets" {
  description = "A list of VPC subnet IDs for the Fargate task. Must have outbound internet access."
  type        = list(string)
  # No default, must be provided
}

variable "fargate_security_groups" {
  description = "A list of security group IDs for the Fargate task. Must allow outbound traffic for S3/CW."
  type        = list(string)
  # No default, must be provided
}
