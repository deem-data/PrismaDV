import warnings
from abc import ABC, abstractmethod

# Suppress PySpark deprecation warnings before importing PySpark
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")
warnings.filterwarnings(
    "ignore", message=".*distutils Version classes are deprecated.*"
)
warnings.filterwarnings("ignore", message=".*is_datetime64tz_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*use of fork\\(\\) may lead to deadlocks.*")

import numpy as np
import pydeequ
import os
from pyspark.sql import SparkSession

# Detect if running in cluster environment (SLURM)
_is_cluster = os.environ.get("SLURM_JOB_ID") is not None

# Get CPU count: prefer SLURM allocated CPUs over system total
if _is_cluster:
    # SLURM: use allocated CPUs (prefer job-specific variables)
    # Order: SLURM_CPUS_PER_TASK (most specific) > SLURM_JOB_CPUS_PER_NODE (job allocation) > SLURM_CPUS_ON_NODE (fallback)
    slurm_cpus = (
        os.environ.get("SLURM_CPUS_PER_TASK")
        or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
        or os.environ.get("SLURM_CPUS_ON_NODE")
    )
    cores = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 4)
else:
    # Local: use system CPU count
    cores = os.cpu_count() or 4

# Safety check: ensure we don't use too many CPUs
assert cores < 20, f"CPU count ({cores}) exceeds safety limit of 20. We might set cpu_count to a wrong value."

# Set shuffle partitions: min 8, max 32, default 2x cores
# AQE will optimize partition sizes automatically, so this is just a starting point
shuffle_parts = min(max(8, cores * 2), 32)


class AbstractDataQualityManager(ABC):
    """
    Abstract base class for managing data quality operations using different backends.
    """

    @staticmethod
    def spark_df_from_pandas_df(pandas_df, schema=None, spark_session=None):
        import logging
        import sys
        from io import StringIO

        # Suppress Spark logging and redirect output during SparkSession creation
        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger("pyspark").setLevel(logging.ERROR)
        logging.getLogger("org.apache.spark").setLevel(logging.ERROR)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            # Adjust configuration based on environment
            if _is_cluster:
                # Cluster environment: optimize for performance
                master = f"local[{cores}]"  # Explicit core count for better control
                driver_memory = "4g"  # Match local memory for better performance
                cluster_shuffle_parts = shuffle_parts  # Use same as local
                # Optimize Arrow for cluster (larger batches, better performance)
                arrow_enabled = "true"
                arrow_max_records = "10000"  # Larger batches for better throughput
            else:
                # Local development: use all cores, more memory
                master = f"local[*]"
                driver_memory = "4g"
                cluster_shuffle_parts = shuffle_parts
                arrow_enabled = "true"
                arrow_max_records = "10000"

            # Build common configs
            spark_builder = (
                SparkSession.builder.appName("deequ-local")
                .master(master)
                .config("spark.driver.memory", driver_memory)
                .config("spark.sql.shuffle.partitions", str(cluster_shuffle_parts))
                .config("spark.default.parallelism", str(cluster_shuffle_parts))
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.adaptive.skewJoin.enabled", "true")
                .config(
                    "spark.serializer", "org.apache.spark.serializer.KryoSerializer"
                )
                .config("spark.sql.execution.arrow.pyspark.enabled", arrow_enabled)
                .config("spark.sql.execution.arrow.maxRecordsPerBatch", arrow_max_records)
                .config("spark.jars.ivy.log", "none")
                .config("spark.hadoop.native.lib", "false")
                .config("spark.jars.packages", pydeequ.deequ_maven_coord)
                .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
                .config("spark.driver.host", "localhost")
                .config("spark.ui.showConsoleProgress", "false")
            )

            # Add cluster-specific optimizations
            if _is_cluster:
                # Optimize GC for cluster (G1GC is better for larger heaps)
                gc_options = (
                    "-XX:+UseG1GC "
                    "-XX:MaxGCPauseMillis=200 "
                    "-XX:+UseStringDeduplication "
                    "-Dlog4j.logger.org.apache.ivy=ERROR "
                    "-Dlog4j.logger.org.apache.spark=ERROR"
                )
            else:
                gc_options = (
                    "-Dlog4j.logger.org.apache.ivy=ERROR "
                    "-Dlog4j.logger.org.apache.spark=ERROR"
                )

            if spark_session is None:
                spark = spark_builder.config("spark.driver.extraJavaOptions", gc_options).getOrCreate()
                spark.sparkContext.setLogLevel("ERROR")
            else:
                spark = spark_session
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Convert NaN to None for Spark compatibility
        pandas_df = pandas_df.replace(np.nan, None)

        if schema is not None:
            spark_df = spark.createDataFrame(pandas_df, schema=schema)
        else:
            spark_df = spark.createDataFrame(pandas_df)

        return spark_df, spark

    @abstractmethod
    def validate_on_spark_df(
        self, spark, spark_df, code_list_for_constraints, return_raw=False
    ):
        """
        Validate the Spark DataFrame using the provided constraints.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def filter_valid_constraints_on_spark(
        self, code_list_for_constraints, spark, spark_df
    ) -> list:
        """
        Filter out invalid constraints from the provided list.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    @abstractmethod
    def build_validation_results(
        code_list_for_constraints, status, valid_code_column_map
    ):
        """
        Build validation results based on the constraints and their statuses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
