import logging
from src.core.meters import AverageEpochMeter


def build() -> dict:

    loss_meters = {
        "main_loss": AverageEpochMeter("Main Loss"),
        "total_loss": AverageEpochMeter("Total Loss"),
        "spatial_location_loss": AverageEpochMeter("Spatial Location Loss"),
        "temporal_location_loss": AverageEpochMeter("Temporal Location Loss"),
        "aux_classification_loss": AverageEpochMeter("Aux Classification Loss"),
        "prototype_loss": AverageEpochMeter("Prototypical Loss"),
    }

    return loss_meters
