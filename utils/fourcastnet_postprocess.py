def get_path_from_yaml(yaml_file, target_string="time_means_path:"):
    matching_lines = []

    with open(yaml_file, "r") as file:
        for line in file:
            if target_string in line:
                matching_lines.append(line)

    output_path = matching_lines[0].split(":")[-1].split("'")[-2]
    return output_path

