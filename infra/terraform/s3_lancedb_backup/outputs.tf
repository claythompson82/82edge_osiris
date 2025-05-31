output "backup_s3_bucket_name" {
  description = "The name of the S3 bucket created for LanceDB backups."
  value       = aws_s3_bucket.backup_bucket.bucket
}

output "backup_s3_bucket_arn" {
  description = "The ARN of the S3 bucket created for LanceDB backups."
  value       = aws_s3_bucket.backup_bucket.arn
}

output "backup_task_definition_arn" {
  description = "The ARN of the ECS Task Definition created for the backup job."
  value       = aws_ecs_task_definition.backup_task_def.arn
}

output "backup_schedule_rule_name" {
  description = "The name of the CloudWatch Event Rule used for scheduling backups."
  value       = aws_cloudwatch_event_rule.backup_schedule_rule.name
}

output "backup_log_group_name" {
  description = "The name of the CloudWatch Log Group for the backup task."
  value       = aws_cloudwatch_log_group.backup_log_group.name
}
