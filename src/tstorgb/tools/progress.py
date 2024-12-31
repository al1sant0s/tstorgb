def report_progress(prefix_str, parsing_info):
    # Clear line.
    print(150 * " ", end="\r")
    # Print progress.
    print(f"{prefix_str} {parsing_info}", end="\r")
