import torch
import math
from parameter import *


EARTH_MEAN_RADIUS_METER = 6371000.0


class Point:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng


def haversine_distance(point1: Point, point2: Point):
    """

    @param point1:
    @param point2:
    @return: distance (meter)
    """
    def same_coords(point_a, point_b):
        return point_a.lat == point_b.lat and point_a.lng == point_b.lng

    if torch.is_tensor(point1.lat) and torch.is_tensor(point1.lng) and \
            torch.is_tensor(point2.lat) and torch.is_tensor(point2.lng):
        if same_coords(point1, point2):
            return torch.tensor([0.], device=DEVICE)

        delta_lat = torch.deg2rad(point2.lat - point1.lat)
        delta_lng = torch.deg2rad(point2.lng - point1.lng)
        v0 = torch.sin(delta_lat / 2.0) * torch.sin(delta_lat / 2.0)
        v1 = torch.sin(delta_lng / 2.0) * torch.sin(delta_lng / 2.0)
        v2 = torch.cos(torch.deg2rad(point1.lat)) * torch.cos(torch.deg2rad(point2.lat))
        h = v0 + v1 * v2
        c = torch.atan2(torch.sqrt(h), torch.sqrt(- h + 1)) * 2.0
        d = c * EARTH_MEAN_RADIUS_METER

    else:
        if same_coords(point1, point2):
            return 0.0
        delta_lat = math.radians(point2.lat - point1.lat)
        delta_lng = math.radians(point2.lng - point1.lng)
        h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(point1.lat)) * math.cos(
            math.radians(point2.lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
        c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
        d = EARTH_MEAN_RADIUS_METER * c

    return d


if __name__ == '__main__':
    print(haversine_distance(Point(1, 1), Point(1, 2)))
