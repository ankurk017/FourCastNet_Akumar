import numpy as np
import xarray as xr
import pandas as pd
import os
import datetime
from collections import ChainMap
from utils.fourcastnet_postprocess import get_path_from_yaml


def create_nc_from_h5_fourcastnet(
    input_file="test_folder/autoregressive_predictions_2020C4Laura01_u10_vis.h5",
    input_yaml="test.yaml",
    output_file="output.nc",
):
    time_means = get_path_from_yaml(input_yaml, "time_means_path")
    global_means = np.load(get_path_from_yaml(input_yaml, "global_means_path"))
    global_stds = np.load(get_path_from_yaml(input_yaml, "global_stds_path"))
    lon_path = get_path_from_yaml(input_yaml, "lons_path")
    lat_path = get_path_from_yaml(input_yaml, "lats_path")
    var_path = get_path_from_yaml(input_yaml, "vars_path")
    check_point_path = get_path_from_yaml(input_yaml, "pretrained_ckpt_path")
    start_date = get_path_from_yaml(input_yaml, "inference_start_date")
    freq = get_path_from_yaml(input_yaml, "inference_freq")
    model_name = get_path_from_yaml(input_yaml, "model_name")

    var_list = np.load(var_path, allow_pickle=True).item()

    start_datetime = pd.to_datetime(
        start_date, format="%Y%m%d%H", errors="coerce")

    in_h5 = xr.open_dataset(input_file)
    modified_dataset = in_h5.rename(
        {
            "phony_dim_1": "time",
            "phony_dim_2": "vars",
            "phony_dim_3": "lat",
            "phony_dim_4": "lon",
        }
    )

    num_time_steps = modified_dataset.dims["time"]

    start_datetime = pd.to_datetime(
        start_datetime, format="%Y%m%d%H", errors="coerce")

    date_range = pd.date_range(
        start=start_datetime, periods=modified_dataset.dims["time"], freq=freq
    )

    modified_dataset["time"] = date_range
    modified_dataset["lat"] = np.load(lat_path)
    modified_dataset["lon"] = np.load(lon_path)
    modified_dataset["vars"] = list(var_list.keys())

    modified_dataset = modified_dataset.squeeze("phony_dim_0")

    modified_dataset.attrs["creation_time"] = str(
        datetime.datetime.now())+' UTC'
    modified_dataset.attrs["model_name"] = model_name
    modified_dataset.attrs["initial_condition"] = str(
        modified_dataset["time"].isel(time=0).values
    ).replace("T", " ")+' UTC'

    modified_dataset["ground_truth"] = xr.DataArray(
        (modified_dataset["ground_truth"].values * global_stds) + global_means,
        dims=modified_dataset["ground_truth"].dims,
        coords=modified_dataset["ground_truth"].coords,
    )
    modified_dataset["predicted"] = xr.DataArray(
        (modified_dataset["predicted"].values * global_stds) + global_means,
        dims=modified_dataset["predicted"].dims,
        coords=modified_dataset["predicted"].coords,
    )

    modified_dataset.to_netcdf(output_file)

    var_path = get_path_from_yaml(input_yaml, "vars_path")
    fourcastnet_sample = modified_dataset

    for variable_listed in ('predicted', 'ground_truth'):
        var_data = [
            {variable: fourcastnet_sample[variable_listed].sel(
                vars=variable).drop("vars")}
            for variable in fourcastnet_sample.vars.values
        ]
        predicted = xr.Dataset(dict(ChainMap(*var_data)))

        predicted.attrs = fourcastnet_sample.attrs
        for var_name in list(predicted.variables.keys())[3:]:
            predicted[var_name].attrs = np.load(
                var_path, allow_pickle=True).item()[var_name]

        predicted.to_netcdf(output_file.split('.')[0].replace(
            'autoregressive_predictions_', '')+f'_{variable_listed}.nc')
