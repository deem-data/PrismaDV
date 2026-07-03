import warnings
import sys
import os
import logging

from prismadv.utils import load_dotenv

load_dotenv()

# Suppress deprecation warnings globally for all prismadv imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyspark")
warnings.filterwarnings("ignore", message=".*distutils Version classes are deprecated.*")
warnings.filterwarnings("ignore", message=".*is_datetime64tz_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*use of fork\\(\\) may lead to deadlocks.*")
warnings.simplefilter("ignore", DeprecationWarning)

# Override warnings.showwarning to suppress deprecation warnings
def _suppress_deprecation_warning(message, category, filename, lineno, file=None, line=None):
    if category == DeprecationWarning:
        return
    if file is None:
        file = sys.stderr
    warnings._showwarning_orig(message, category, filename, lineno, file, line)

warnings._showwarning_orig = warnings.showwarning
warnings.showwarning = _suppress_deprecation_warning

# Configure logging globally for all prismadv imports
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
logging.getLogger("org.apache.spark").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.INFO)
logging.basicConfig(level=logging.WARNING, force=True)

# Set Spark environment variables globally
os.environ["SPARK_JARS_IVY_LOG"] = "none"
os.environ["SPARK_UI_SHOW_CONSOLE_PROGRESS"] = "false"