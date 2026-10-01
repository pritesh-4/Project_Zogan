"""
=============================================================================
🐘 PROJECT ZOGAN — GEOFENCING & GEOLOCATION UTILITIES (PHASE 5)
=============================================================================

This module provides software-based geographic utilities for calculating
real-world geodesic distances, testing zone memberships (point-in-zone),
evaluating boundary distances, and classifying areas (Forest, Buffer, Village).

Pipeline Integration:
  Camera / Sim Coordinates + Zone Definitions -> ai/geofence.py -> ai/risk_engine.py

Key Capabilities:
  - Haversine geodesic distance calculation (accurate meters on WGS84 sphere)
  - Concentric and polygonal zone geometry support
  - Point-in-zone detection (radial circle & Jordan curve ray casting)
  - Boundary distance and proximity detection
  - Multi-zone priority classification (Village > Buffer > Forest > Outside)

⚠️ IMPORTANT CONSTRAINTS & LIMITATIONS:
  - Software calculations only (no physical GPS hardware attached in Phase 5).
  - Camera location does NOT equal elephant location.
  - Distances are geodesic surface distances, not terrain-elevated distances.
=============================================================================
"""

import math
from typing import List, Optional, Tuple

# Earth mean radius in meters (WGS84 spherical approximation)
EARTH_RADIUS_METERS: float = 6371000.0

# Zone Type Identifiers
ZONE_TYPE_PROTECTED = "VILLAGE"
ZONE_TYPE_WARNING = "BUFFER"
ZONE_TYPE_MONITORING = "FOREST"
ZONE_TYPE_OUTSIDE = "OUTSIDE"

ZONE_PRIORITY_ORDER = [
    ZONE_TYPE_PROTECTED,
    ZONE_TYPE_WARNING,
    ZONE_TYPE_MONITORING,
]


# =============================================================================
# 📐 GEODESIC DISTANCE & BEARING CALCULATIONS
# =============================================================================


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle surface distance between two points in meters
    using the Haversine formula.

    Inputs:
      lat1, lon1: Coordinates of point 1 in decimal degrees
      lat2, lon2: Coordinates of point 2 in decimal degrees

    Returns:
      Distance in meters (float >= 0.0)
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    sin_dphi = math.sin(delta_phi / 2.0)
    sin_dlambda = math.sin(delta_lambda / 2.0)

    a = sin_dphi * sin_dphi + math.cos(phi1) * math.cos(phi2) * sin_dlambda * sin_dlambda
    # Clamp 'a' to [0.0, 1.0] to safeguard against numerical precision errors
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_METERS * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the initial compass bearing (azimuth) from point 1 to point 2.

    Returns:
      Bearing in degrees [0.0, 360.0), where 0 = North, 90 = East, etc.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    initial_bearing = math.atan2(y, x)
    degrees = (math.degrees(initial_bearing) + 360.0) % 360.0
    return degrees


# =============================================================================
# 🗺️ GEOGRAPHIC ZONE DATA MODEL
# =============================================================================


class GeoZone:
    """
    Represents a designated geographic zone (e.g. Village Settlement, Buffer, Forest).
    Supports circular (radial) and polygonal boundary representations.
    """

    def __init__(
        self,
        name: str,
        zone_type: str,
        geometry_type: str = "circle",
        center: Optional[Tuple[float, float]] = None,
        radius_meters: Optional[float] = None,
        polygon_points: Optional[List[Tuple[float, float]]] = None,
        description: str = "",
        priority: int = 1,
    ):
        self.name = name
        self.zone_type = zone_type.upper()
        self.geometry_type = geometry_type.lower()
        self.center = center  # (lat, lon)
        self.radius_meters = float(radius_meters) if radius_meters is not None else 0.0
        self.polygon_points = polygon_points or []  # List of (lat, lon)
        self.description = description
        self.priority = priority

        if self.geometry_type == "circle" and (self.center is None or self.radius_meters <= 0):
            raise ValueError(f"Circle zone '{name}' requires valid center (lat, lon) and positive radius_meters.")
        if self.geometry_type == "polygon" and len(self.polygon_points) < 3:
            raise ValueError(f"Polygon zone '{name}' requires at least 3 vertex coordinates.")

    def is_inside(self, lat: float, lon: float) -> bool:
        """
        Determines whether the given coordinate point falls inside this zone.
        """
        if self.geometry_type == "circle":
            dist = calculate_distance(lat, lon, self.center[0], self.center[1])
            return dist <= self.radius_meters

        elif self.geometry_type == "polygon":
            # Ray casting algorithm (Jordan curve theorem)
            # x = lon, y = lat
            n = len(self.polygon_points)
            inside = False
            p1_lat, p1_lon = self.polygon_points[0]
            for i in range(1, n + 1):
                p2_lat, p2_lon = self.polygon_points[i % n]
                if (lat > min(p1_lat, p2_lat)) and (lat <= max(p1_lat, p2_lat)):
                    if lon <= max(p1_lon, p2_lon):
                        if p1_lat != p2_lat:
                            x_inters = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                        if p1_lon == p2_lon or lon <= x_inters:
                            inside = not inside
                p1_lat, p1_lon = p2_lat, p2_lon
            return inside

        return False

    def distance_to(self, lat: float, lon: float) -> float:
        """
        Calculates distance in meters from the given point to the zone boundary.
        If the point is INSIDE the zone, distance is 0.0.
        """
        if self.is_inside(lat, lon):
            return 0.0

        if self.geometry_type == "circle":
            dist_to_center = calculate_distance(lat, lon, self.center[0], self.center[1])
            return max(0.0, dist_to_center - self.radius_meters)

        elif self.geometry_type == "polygon":
            # Minimum distance to any boundary edge
            min_dist = float("inf")
            n = len(self.polygon_points)
            for i in range(n):
                p1 = self.polygon_points[i]
                p2 = self.polygon_points[(i + 1) % n]
                d = _distance_point_to_segment(lat, lon, p1, p2)
                if d < min_dist:
                    min_dist = d
            return min_dist

        return 0.0

    def is_near_boundary(self, lat: float, lon: float, threshold_meters: float = 50.0) -> bool:
        """
        Returns True if the point is outside the zone but within threshold_meters of its border.
        """
        d = self.distance_to(lat, lon)
        return 0.0 < d <= threshold_meters

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "zone_type": self.zone_type,
            "geometry_type": self.geometry_type,
            "center": self.center,
            "radius_meters": self.radius_meters,
            "priority": self.priority,
            "description": self.description,
        }


def _distance_point_to_segment(
    lat: float,
    lon: float,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """
    Approximates perpendicular distance from point (lat, lon) to geodesic segment p1-p2.
    """
    d1 = calculate_distance(lat, lon, p1[0], p1[1])
    d2 = calculate_distance(lat, lon, p2[0], p2[1])
    seg_len = calculate_distance(p1[0], p1[1], p2[0], p2[1])

    if seg_len <= 1e-6:
        return d1

    # Project point onto segment in approximate local Cartesian frame
    # Convert lat/lon offsets to meters relative to p1
    lat_mid = (p1[0] + p2[0]) / 2.0
    m_per_deg_lat = 111132.95
    m_per_deg_lon = 111412.84 * math.cos(math.radians(lat_mid))

    v_x = (p2[1] - p1[1]) * m_per_deg_lon
    v_y = (p2[0] - p1[0]) * m_per_deg_lat

    w_x = (lon - p1[1]) * m_per_deg_lon
    w_y = (lat - p1[0]) * m_per_deg_lat

    c1 = w_x * v_x + w_y * v_y
    if c1 <= 0:
        return d1
    c2 = v_x * v_x + v_y * v_y
    if c2 <= c1:
        return d2

    b = c1 / c2
    proj_lon = p1[1] + (b * (p2[1] - p1[1]))
    proj_lat = p1[0] + (b * (p2[0] - p1[0]))
    return calculate_distance(lat, lon, proj_lat, proj_lon)


# =============================================================================
# 🔍 ZONE QUERY & CLASSIFICATION FUNCTIONS
# =============================================================================


def is_point_in_zone(lat: float, lon: float, zone: GeoZone) -> bool:
    """
    Determines whether a coordinate point is inside the specified zone.
    """
    return zone.is_inside(lat, lon)


def distance_to_zone(lat: float, lon: float, zone: GeoZone) -> float:
    """
    Returns distance in meters from coordinate point to the specified zone boundary.
    Returns 0.0 if inside.
    """
    return zone.distance_to(lat, lon)


def classify_zone(lat: float, lon: float, zones: List[GeoZone]) -> str:
    """
    Classifies a coordinate point into one of the configured zone types.
    Zones are evaluated in priority order (Village/Protected > Buffer/Warning > Forest/Monitoring).
    If outside all zones, returns 'OUTSIDE'.
    """
    sorted_zones = sorted(zones, key=lambda z: z.priority, reverse=True)
    for zone in sorted_zones:
        if zone.is_inside(lat, lon):
            return zone.zone_type

    return ZONE_TYPE_OUTSIDE


def get_distance_to_protected_zone(lat: float, lon: float, zones: List[GeoZone]) -> float:
    """
    Finds the distance in meters to the nearest PROTECTED / VILLAGE zone.
    """
    protected_zones = [z for z in zones if z.zone_type in (ZONE_TYPE_PROTECTED, "PROTECTED")]
    if not protected_zones:
        return float("inf")

    return min(z.distance_to(lat, lon) for z in protected_zones)


def create_default_zones(
    village_center: Tuple[float, float] = (20.119000, 85.119000),
    village_radius: float = 300.0,
    buffer_radius: float = 800.0,
    forest_radius: float = 2000.0,
) -> List[GeoZone]:
    """
    Generates standard three-tier concentric wildlife monitoring zones:
      1. Village (Protected settlement, priority 3)
      2. Buffer (Warning zone, priority 2)
      3. Forest (Monitoring area, priority 1)
    """
    village_zone = GeoZone(
        name="Village Settlement Zone",
        zone_type=ZONE_TYPE_PROTECTED,
        geometry_type="circle",
        center=village_center,
        radius_meters=village_radius,
        description="High-priority human settlement and agriculture perimeter",
        priority=3,
    )

    buffer_zone = GeoZone(
        name="Buffer Warning Zone",
        zone_type=ZONE_TYPE_WARNING,
        geometry_type="circle",
        center=village_center,
        radius_meters=buffer_radius,
        description="Intermediate warning corridor between forest and village",
        priority=2,
    )

    forest_zone = GeoZone(
        name="Forest Monitoring Zone",
        zone_type=ZONE_TYPE_MONITORING,
        geometry_type="circle",
        center=village_center,
        radius_meters=forest_radius,
        description="Natural wildlife forest habitat monitoring boundary",
        priority=1,
    )

    return [village_zone, buffer_zone, forest_zone]
