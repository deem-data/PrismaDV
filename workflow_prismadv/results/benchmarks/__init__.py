"""Units and ground truth of each benchmark, and where a method's files for a unit live. No method logic here.

Every reader in workflow_prismadv/results/methods returns outcome rows with these columns.
"""
COLUMNS = ["benchmark", "dataset", "unit", "arm", "rep", "safe", "alarm"]
