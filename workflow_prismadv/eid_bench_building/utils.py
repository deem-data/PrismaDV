def process_bad_assertions(bad_assertions, assertions):
    new_assertions = assertions.copy()
    assertions_to_remove = []

    for bad_assertion in bad_assertions['result']:
        assertion_index = bad_assertion['bad_assertion_id']
        decided_to_keep = bad_assertion['decided_to_keep']

        if decided_to_keep is False:
            assertions_to_remove.append(assertion_index)
            print("assertions_removed:\n"
                  f"{new_assertions[assertion_index]['code']}")
        else:
            if bad_assertion['suggested_new_assertion'].startswith("# ASSERTION_START"):
                lines = bad_assertion['suggested_new_assertion'].split("\n")
                bad_assertion['suggested_new_assertion'] = "\n".join(lines[1:])
            if bad_assertion['suggested_new_assertion'].rstrip().endswith("# ASSERTION_END"):
                lines = bad_assertion['suggested_new_assertion'].split("\n")
                bad_assertion['suggested_new_assertion'] = "\n".join(lines[:-1]) + "\n"
            if bad_assertion['suggested_new_assertion'].startswith("# Assertion"):
                lines = bad_assertion['suggested_new_assertion'].split("\n")
                bad_assertion['suggested_new_assertion'] = "\n".join(lines[1:])
            print("assertions_updated:\n"
                  f"{new_assertions[assertion_index]['code']}\n"
                  f"=>\n"
                  f"{bad_assertion['suggested_new_assertion']}")
            new_assertions[assertion_index]['code'] = bad_assertion['suggested_new_assertion']

    # Remove unwanted assertions by index
    new_assertions = [
        assertion for idx, assertion in enumerate(new_assertions)
        if idx not in assertions_to_remove
    ]
    num_removed = len(assertions_to_remove)
    num_modified = len(bad_assertions['result']) - len(assertions_to_remove)
    return new_assertions, num_removed, num_modified
