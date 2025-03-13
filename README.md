# Accelerating Fraud Detection in Financial Services with NVIDIA RAPIDS on AWS

This repository demonstrates how to accelerate fraud detection workflows in financial services using **NVIDIA RAPIDS** on **AWS**. The project showcases a GPU-accelerated data pipeline that significantly improves processing speed and cost efficiency compared to traditional CPU-based workflows.

## Table of Contents
1. [What is NVIDIA RAPIDS?](#what-is-nvidia-rapids)
2. [Why NVIDIA RAPIDS Benefits Financial Services](#why-nvidia-rapids-benefits-financial-services)
3. [Setting Up EMR Clusters with NVIDIA GPUs](#setting-up-emr-clusters-with-nvidia-gpus)
4. [Fraud Detection Pipeline](#fraud-detection-pipeline)
5. [Performance Benchmarks and Cost Efficiency](#performance-benchmarks-and-cost-efficiency)
6. [Conclusion](#conclusion)

---

## What is NVIDIA RAPIDS?
NVIDIA RAPIDS is a suite of open-source GPU-accelerated libraries designed to accelerate data science and machine learning workflows. Built on CUDA-X AI, RAPIDS integrates seamlessly with popular frameworks like Apache Spark, enabling massive parallelization of data processing tasks. By leveraging the computational power of GPUs, RAPIDS drastically reduces the time required for operations such as data ingestion, feature engineering, and model training.

Unlike traditional CPU-based workflows, RAPIDS processes large datasets in-memory, bypassing bottlenecks associated with disk I/O and CPU thread limitations. This makes it ideal for real-time analytics and scalable pipelines, particularly in industries like financial services where speed and accuracy are critical.

## Why NVIDIA RAPIDS Benefits Financial Services
Fraud detection in financial services is challenging, requiring real-time insights from complex, large-scale datasets. Traditional CPU-based systems struggle to keep pace with the sheer volume of transactions and the computational demands of feature engineering. NVIDIA RAPIDS, with its GPU-accelerated capabilities, enables faster and more cost-effective data processing, providing an edge in building scalable and efficient data pipelines.

With RAPIDS, financial institutions can:
- **Detect fraud in near real-time**, minimizing losses.
- **Scale efficiently** to handle growing transaction volumes.
- **Lower infrastructure costs** by reducing reliance on expensive CPU clusters.

This repository demonstrates a GPU-accelerated fraud detection workflow using NVIDIA RAPIDS and Apache Spark on AWS. The workflow showcases significant improvements in processing speed and cost efficiency compared to CPU-based alternatives.

---

## Setting Up EMR Clusters with NVIDIA GPUs
To leverage NVIDIA RAPIDS for fraud detection, the EMR cluster must be configured with GPU-enabled instances and specific settings to optimize performance.
You can download the source files for [customers](https://d2908q01vomqb2.cloudfront.net/artifacts/DBSBlogs/FSI-NVIDIA-rapids/customers_parquet.tar.gz), [transactions](https://d2908q01vomqb2.cloudfront.net/artifacts/DBSBlogs/FSI-NVIDIA-rapids/transactions_parquet.tar.gz), [terminals](https://d2908q01vomqb2.cloudfront.net/artifacts/DBSBlogs/FSI-NVIDIA-rapids/terminals_parquet.tar.gz)

### GPU Cluster Configuration
#### Instances:
- **Primary Node:** `M5.xlarge`
- **Core Nodes:** `12 nodes of G6.4xlarge` (GPU-enabled)

#### Bootstrap Script:
```bash
#!/bin/bash
set -ex
sudo mkdir -p /spark-rapids-cgroup/devices
sudo mount -t cgroup -o devices cgroupv1-devices /spark-rapids-cgroup/devices
sudo chmod a+rwx -R /spark-rapids-cgroup
sudo pip3 install numpy
```

#### JSON Configuration:
```json
[
  {
    "Classification": "spark",
    "Properties": {
      "enableSparkRapids": "true"
    }
  },
  {
    "Classification": "spark-defaults",
    "Properties": {
      "spark.executor.memory": "30G",
      "spark.executor.instances": "12",
      "spark.executor.resource.gpu.amount": "1",
      "spark.plugins": "com.nvidia.spark.SQLPlugin",
      "spark.rapids.sql.enabled": "true"
    }
  }
]
```

---

## Fraud Detection Pipeline
### Step 1: Initialize Spark Session with GPU Optimizations
```python
spark = SparkSession.builder \
    .config("spark.executor.memory", "80G") \
    .config("spark.sql.shuffle.partitions", "20000") \
    .getOrCreate()
```

### Step 2: Load and Prepare Data
```python
customers_df = spark.read.parquet(customers_path).repartition(300)
transactions_df = spark.read.parquet(transactions_path).repartition(1000)
```

### Step 3: Convert TX_DATETIME to Timestamp
```python
transactions_df = transactions_df.withColumn("TX_DATETIME", F.col("TX_DATETIME").cast("timestamp"))
```

### Step 4: Extract Date Components
```python
transactions_df = transactions_df.withColumn("yyyy", year(F.col("TX_DATETIME"))) \
                                 .withColumn("mm", month(F.col("TX_DATETIME"))) \
                                 .withColumn("dd", dayofmonth(F.col("TX_DATETIME")))
```

### Step 5: Save Final Dataset
```python
final_df.write.mode("overwrite").parquet("s3://path/to/output/")
```

---

## Performance Benchmarks and Cost Efficiency
| Instance Type | Core Count | Hourly Cost | Run Time (Minutes) | Total Cost |
|--------------|------------|-------------|--------------------|------------|
| GPU (G6.4xlarge) | 12 | $1.323 | 43 | $11.52 |
| CPU (R7i.4xLarge) | 12 | $1.058 | 450 | $96.66 |
| CPU (R7a.4xlarge) | 12 | $1.217 | 246 | $60.67 |

### Key Takeaways:
- **up to 10.5x Speedup**: GPU workflows process data in minutes, enabling real-time fraud alerts.
- **up to 8.4x Cost Reduction**: Lower infrastructure costs due to reduced runtime and optimized resource usage.

---

## Conclusion
NVIDIA RAPIDS, integrated with AWS, transforms fraud detection pipelines by delivering unmatched performance and cost efficiency. By combining GPU acceleration with AWS’s scalable infrastructure, businesses can achieve real-time insights, minimize losses, and build a future-proof fraud detection system.

**Ready to supercharge your pipeline?** Explore NVIDIA RAPIDS and AWS GPU instances today to unlock the next level of speed and efficiency.

