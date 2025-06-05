# Production Data Strategy: LanceDB & Redis

This document describes how to operate Osiris database components in a production
environment. LanceDB is used by the `llm_sidecar` to store proposal logs, while
Redis backs the tick bus and can be used for caching or messaging.

## LanceDB High Availability
LanceDB does not currently provide built‑in clustering. Treat it similar to a
single‑node database and run it on resilient storage:

- Deploy the LanceDB container as a **StatefulSet** with a PersistentVolume
  backed by a highly available storage class (e.g. AWS EBS with replication or
  an NFS/EFS share).
- Only one instance should write to the dataset. If multiple replicas are
  required, run an active/passive setup with a readiness probe so only the leader
  receives traffic.
- Store the underlying data on an object store or shared filesystem when
  possible so a standby node can mount the same files if a failover occurs.

## Redis High Availability
For production clusters use a Redis deployment that provides automatic failover.
Common options include:

- **Redis Sentinel** – a primary/replica setup with sentinels to handle
  promotion. Charts like `bitnami/redis` support this via `architecture=replication`.
- **Redis Cluster** – shards data across multiple nodes. Use the Bitnami chart or
  an operator such as `spotahome/redis-operator` to manage the cluster.
- Always enable persistence (RDB or AOF) and use PersistentVolumes for each pod.

## LanceDB Backup & Recovery
- The LanceDB dataset lives at `/app/lancedb_data` in the sidecar container.
- Schedule a `CronJob` or external task to run `aws s3 sync` (or `rclone`) from
  this directory to durable object storage. The provided Terraform module under
  `infra/terraform/s3_lancedb_backup` is an example using ECS Fargate.
- To restore, copy the backup files back into the volume before starting the
  sidecar or attach the object store path if LanceDB supports it.

## Redis Backup & Recovery
- If persistence is enabled, trigger a snapshot with `redis-cli BGSAVE` and copy
  `dump.rdb` (or `appendonly.aof` when using AOF) from the data directory.
- Automate this with a `CronJob` that archives the file to S3 or another backup
  location.
- To recover, place the saved RDB/AOF file in the Redis data directory before
  starting the server.

For general operational tips see [Day 2 Operations](DAY2_OPERATIONS.md).
