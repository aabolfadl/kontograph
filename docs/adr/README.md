# Architecture Decision Records

Six short records explaining **why not**. Rejected alternatives say more about engineering judgment
than chosen ones do, because anyone can list what they used.

Format: context → decision → consequences → *what would change our mind*. That last section is the
one that matters: a decision with no falsifier is a preference.

| # | Decision | Rejected |
|---|---|---|
| [0001](0001-redpanda-over-kafka.md) | Redpanda as the log | Kafka + Confluent Schema Registry |
| [0003](0003-not-spark-structured-streaming.md) | Not Spark Structured Streaming | Spark as the single engine |
| [0004](0004-iceberg-sqlite-catalog.md) | Iceberg with a SQLite catalog | REST catalog service; Delta; plain Parquet |
| [0009](0009-content-hashed-datasets.md) | Content-hashed dataset manifests | DVC |
| [0010](0010-compose-now-k8s-later.md) | Docker Compose as the deployment target | Kubernetes and Terraform at this stage |
| [0011](0011-native-quality-gate.md) | Native quality gate on the blocking path | Great Expectations inline |
