# BigQuery Geospatial Reference

Complete guide to geospatial functions and analysis in BigQuery.

## Geography Data Types

BigQuery supports the `GEOGRAPHY` type for geospatial data.

### Creating Geography Objects

```sql
-- Point from coordinates (longitude, latitude)
SELECT ST_GEOGPOINT(-122.4194, 37.7749) AS san_francisco;

-- From Well-Known Text (WKT)
SELECT ST_GEOGFROMTEXT('POINT(-122.4194 37.7749)') AS point;
SELECT ST_GEOGFROMTEXT('LINESTRING(0 0, 1 1, 2 0)') AS line;
SELECT ST_GEOGFROMTEXT('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))') AS polygon;

-- From GeoJSON
SELECT ST_GEOGFROMGEOJSON('{"type":"Point","coordinates":[-122.4194,37.7749]}');

-- From WKB (Well-Known Binary)
SELECT ST_GEOGFROMWKB(wkb_column) FROM table;
```

### Geography Constructors

| Function | Description | Example |
|----------|-------------|---------|
| `ST_GEOGPOINT(lng, lat)` | Create point | `ST_GEOGPOINT(-122, 37)` |
| `ST_MAKELINE(points)` | Create line from points | `ST_MAKELINE([p1, p2, p3])` |
| `ST_MAKEPOLYGON(ring)` | Create polygon from ring | `ST_MAKEPOLYGON(line)` |
| `ST_GEOGFROMTEXT(wkt)` | Parse WKT | `ST_GEOGFROMTEXT('POINT(0 0)')` |
| `ST_GEOGFROMGEOJSON(json)` | Parse GeoJSON | `ST_GEOGFROMGEOJSON(json_col)` |

## Measurement Functions

### Distance

```sql
-- Distance in meters
SELECT ST_DISTANCE(
  ST_GEOGPOINT(-122.4194, 37.7749),  -- San Francisco
  ST_GEOGPOINT(-118.2437, 34.0522)   -- Los Angeles
) AS distance_meters;
-- Returns: ~559,044 meters

-- Distance in kilometers
SELECT ST_DISTANCE(point_a, point_b) / 1000 AS distance_km;

-- Distance in miles
SELECT ST_DISTANCE(point_a, point_b) / 1609.34 AS distance_miles;
```

### Length and Perimeter

```sql
-- Length of a line (meters)
SELECT ST_LENGTH(ST_GEOGFROMTEXT('LINESTRING(0 0, 1 1, 2 0)')) AS length;

-- Perimeter of a polygon (meters)
SELECT ST_PERIMETER(ST_GEOGFROMTEXT('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'));
```

### Area

```sql
-- Area in square meters
SELECT ST_AREA(ST_GEOGFROMTEXT('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))'));

-- Area in square kilometers
SELECT ST_AREA(boundary) / 1000000 AS area_sq_km FROM regions;

-- Area in acres
SELECT ST_AREA(boundary) / 4046.86 AS area_acres FROM parcels;
```

## Spatial Relationships

### Containment and Intersection

```sql
-- Point in polygon
SELECT ST_CONTAINS(polygon, point);  -- TRUE if point is inside polygon

-- Covers (includes boundary)
SELECT ST_COVERS(polygon, point);  -- TRUE if point is inside or on boundary

-- Intersects
SELECT ST_INTERSECTS(geog1, geog2);  -- TRUE if geometries share any point

-- Disjoint
SELECT ST_DISJOINT(geog1, geog2);  -- TRUE if geometries don't touch

-- Touches (only boundaries touch)
SELECT ST_TOUCHES(geog1, geog2);  -- TRUE if only boundaries meet

-- Within
SELECT ST_WITHIN(point, polygon);  -- TRUE if point is inside polygon
```

### Distance Relationships

```sql
-- Within distance
SELECT ST_DWITHIN(
  store_location,
  customer_location,
  5000  -- 5km in meters
);  -- TRUE if within 5km

-- Find all stores within 10km
SELECT store_name
FROM stores
WHERE ST_DWITHIN(
  location,
  ST_GEOGPOINT(-122.4194, 37.7749),
  10000
);
```

## Spatial Operations

### Intersection and Union

```sql
-- Intersection of two geometries
SELECT ST_INTERSECTION(polygon_a, polygon_b) AS overlap;

-- Union of geometries
SELECT ST_UNION(geog1, geog2) AS combined;

-- Union aggregate (combine many geometries)
SELECT ST_UNION_AGG(boundary) AS merged_boundary
FROM regions
WHERE state = 'California';
```

### Buffers

```sql
-- Create buffer around point (radius in meters)
SELECT ST_BUFFER(
  ST_GEOGPOINT(-122.4194, 37.7749),
  1000  -- 1km radius
) AS buffer_zone;

-- Create buffer around line
SELECT ST_BUFFER(route_line, 100) AS corridor;  -- 100m buffer
```

### Simplification

```sql
-- Simplify geometry (reduce vertices)
SELECT ST_SIMPLIFY(complex_polygon, 100);  -- tolerance in meters

-- Convex hull
SELECT ST_CONVEXHULL(multi_point) AS hull;
```

### Centroid and Boundary

```sql
-- Centroid (center point)
SELECT ST_CENTROID(polygon) AS center;

-- Boundary of polygon
SELECT ST_BOUNDARY(polygon) AS boundary_line;

-- Bounding box
SELECT ST_BOUNDINGBOX(geography) AS bbox;
```

## Accessors

```sql
-- Get coordinates
SELECT ST_X(point) AS longitude;  -- -122.4194
SELECT ST_Y(point) AS latitude;   -- 37.7749

-- Number of points
SELECT ST_NUMPOINTS(line) AS point_count;

-- Check geometry type
SELECT ST_GEOMETRYTYPE(geog) AS geom_type;  -- 'ST_Point', 'ST_Polygon', etc.

-- Check if valid
SELECT ST_ISVALID(geog) AS is_valid;

-- Check if empty
SELECT ST_ISEMPTY(geog) AS is_empty;

-- Dimension
SELECT ST_DIMENSION(geog) AS dim;  -- 0=point, 1=line, 2=polygon
```

## Output Functions

```sql
-- Convert to GeoJSON
SELECT ST_ASGEOJSON(geography) AS geojson;

-- Convert to WKT
SELECT ST_ASTEXT(geography) AS wkt;

-- Convert to WKB
SELECT ST_ASBINARY(geography) AS wkb;
```

## Clustering and Aggregation

### Geographic Clustering

```sql
-- Cluster points by grid
SELECT
  ST_SNAPTOGRID(location, 0.01) AS grid_cell,  -- ~1km grid
  COUNT(*) AS point_count
FROM locations
GROUP BY grid_cell;

-- Geohash clustering
SELECT
  ST_GEOHASH(location, 5) AS geohash,  -- precision 5 (~5km)
  COUNT(*) AS count
FROM locations
GROUP BY geohash;
```

### Aggregate Functions

```sql
-- Collect points into multipoint
SELECT ST_UNION_AGG(location) AS all_points
FROM stores
WHERE region = 'West';

-- Centroid of all points
SELECT ST_CENTROID(ST_UNION_AGG(location)) AS center
FROM stores;

-- Bounding box of all geometries
SELECT ST_BOUNDINGBOX(ST_UNION_AGG(boundary))
FROM regions;
```

## Spatial Joins

### Point in Polygon Join

```sql
SELECT
  c.customer_id,
  r.region_name
FROM customers c
JOIN regions r
  ON ST_CONTAINS(r.boundary, c.location);
```

### Nearest Neighbor

```sql
-- Find nearest store for each customer
SELECT
  c.customer_id,
  (
    SELECT s.store_name
    FROM stores s
    ORDER BY ST_DISTANCE(c.location, s.location)
    LIMIT 1
  ) AS nearest_store
FROM customers c;

-- With distance
SELECT
  c.customer_id,
  s.store_name,
  ST_DISTANCE(c.location, s.location) AS distance_m
FROM customers c
CROSS JOIN stores s
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY c.customer_id
  ORDER BY ST_DISTANCE(c.location, s.location)
) = 1;
```

### K Nearest Neighbors

```sql
SELECT
  customer_id,
  ARRAY_AGG(
    STRUCT(store_name, distance_m)
    ORDER BY distance_m
    LIMIT 3
  ) AS nearest_3_stores
FROM (
  SELECT
    c.customer_id,
    s.store_name,
    ST_DISTANCE(c.location, s.location) AS distance_m
  FROM customers c
  CROSS JOIN stores s
)
GROUP BY customer_id;
```

## Public Datasets

BigQuery has several public geospatial datasets:

```sql
-- US ZIP codes
SELECT * FROM `bigquery-public-data.geo_us_boundaries.zip_codes`;

-- US Census tracts
SELECT * FROM `bigquery-public-data.geo_census_tracts.us_census_tracts_national`;

-- OpenStreetMap
SELECT * FROM `bigquery-public-data.geo_openstreetmap.planet_features`;

-- World country boundaries
SELECT * FROM `bigquery-public-data.geo_international_ports.world_port_index`;
```

## Performance Optimization

### Index Usage

BigQuery automatically indexes GEOGRAPHY columns. Optimize by:

1. Using `ST_DWITHIN` instead of `ST_DISTANCE < threshold`
2. Using `ST_INTERSECTS` with bounding boxes
3. Pre-filtering with geohash

### Pre-filtering Example

```sql
-- Efficient: use spatial predicate
SELECT * FROM locations
WHERE ST_DWITHIN(location, @query_point, 10000);

-- Less efficient: compute all distances then filter
SELECT * FROM locations
WHERE ST_DISTANCE(location, @query_point) < 10000;
```

### Geohash Pre-filter

```sql
-- Pre-filter with geohash before expensive spatial operations
WITH candidates AS (
  SELECT *
  FROM locations
  WHERE ST_GEOHASH(location, 4) IN (
    ST_GEOHASH(@query_point, 4),
    -- Include adjacent cells
    'abc1', 'abc2', 'abc3'
  )
)
SELECT *
FROM candidates
WHERE ST_DWITHIN(location, @query_point, 5000);
```

## Common Patterns

### Service Area Analysis

```sql
-- Find customers within each store's service radius
SELECT
  s.store_id,
  COUNT(c.customer_id) AS customers_in_area
FROM stores s
LEFT JOIN customers c
  ON ST_DWITHIN(s.location, c.location, s.service_radius_m)
GROUP BY s.store_id;
```

### Route Analysis

```sql
-- Calculate total route distance
SELECT
  route_id,
  ST_LENGTH(route_line) AS total_distance_m,
  ST_NUMPOINTS(route_line) AS waypoints
FROM routes;
```

### Hotspot Analysis

```sql
-- Identify dense clusters
SELECT
  ST_GEOHASH(location, 6) AS cell,
  COUNT(*) AS incident_count,
  ST_CENTROID(ST_UNION_AGG(location)) AS cell_center
FROM incidents
GROUP BY cell
HAVING COUNT(*) > 10
ORDER BY incident_count DESC;
```
