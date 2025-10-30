
### Setting Up Monitoring Alerts

Create automated alerts based on AI Gateway inference table data:

```python
def check_endpoint_health(catalog, schema, table_prefix, alert_threshold=0.05):
    """
    Monitor endpoint health and return alert messages for anomalies
    
    Args:
        catalog: Unity Catalog name
        schema: Schema name
        table_prefix: Inference table prefix
        alert_threshold: Error rate threshold (default 5%)
    
    Returns:
        List of alert messages
    """
    alerts = []
    
    # Check recent error rate
    error_stats = spark.sql(f"""
        SELECT 
            COUNT(*) as total_requests,
            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count,
            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) / COUNT(*) as error_rate,
            COUNT(DISTINCT requester) as affected_users
        FROM {catalog}.{schema}.{table_prefix}_payload
        WHERE request_time >= current_timestamp() - INTERVAL 1 HOUR
    """).collect()[0]
    
    if error_stats["error_rate"] > alert_threshold:
        alerts.append(
            f"⚠️ High error rate: {error_stats['error_rate']:.1%} "
            f"({error_stats['error_count']} errors out of {error_stats['total_requests']} requests) "
            f"affecting {error_stats['affected_users']} users"
        )
    
    # Check latency trends
    latency_stats = spark.sql(f"""
        SELECT 
            AVG(execution_duration_ms) as avg_latency,
            PERCENTILE(execution_duration_ms, 0.95) as p95_latency,
            PERCENTILE(execution_duration_ms, 0.99) as p99_latency
        FROM {catalog}.{schema}.{table_prefix}_payload
        WHERE request_time >= current_timestamp() - INTERVAL 1 HOUR
    """).collect()[0]
    
    if latency_stats["p95_latency"] and latency_stats["p95_latency"] > 10000:  # 10 seconds
        alerts.append(
            f"⚠️ High latency detected: P95 = {latency_stats['p95_latency']:.0f}ms, "
            f"P99 = {latency_stats['p99_latency']:.0f}ms, "
            f"Average = {latency_stats['avg_latency']:.0f}ms"
        )
    
    # Check for request volume anomalies
    volume_stats = spark.sql(f"""
        SELECT 
            COUNT(*) as current_hour_requests,
            (SELECT AVG(hourly_requests) 
             FROM (
                 SELECT COUNT(*) as hourly_requests
                 FROM {catalog}.{schema}.{table_prefix}_payload
                 WHERE request_time >= current_timestamp() - INTERVAL 24 HOURS
                 GROUP BY date_trunc('hour', request_time)
             )
            ) as avg_hourly_requests
        FROM {catalog}.{schema}.{table_prefix}_payload
        WHERE request_time >= current_timestamp() - INTERVAL 1 HOUR
    """).collect()[0]
    
    if volume_stats["current_hour_requests"] < volume_stats["avg_hourly_requests"] * 0.5:
        alerts.append(
            f"⚠️ Low request volume: {volume_stats['current_hour_requests']} requests "
            f"(50% below average of {volume_stats['avg_hourly_requests']:.0f})"
        )
    
    # Check for multiple model versions (could indicate deployment issues or A/B testing scenarios)
    version_stats = spark.sql(f"""
        SELECT 
            COUNT(DISTINCT served_entity_id) as active_versions,
            COLLECT_LIST(DISTINCT served_entity_id) as version_list
        FROM {catalog}.{schema}.{table_prefix}_payload
        WHERE request_time >= current_timestamp() - INTERVAL 1 HOUR
    """).collect()[0]
    
    if version_stats["active_versions"] > 1:
        alerts.append(
            f"ℹ️ Multiple model versions active: {version_stats['active_versions']} versions "
            f"({', '.join(version_stats['version_list'])})"
        )
    
    return alerts

# Run health check
alerts = check_endpoint_health(catalog, schema, "scgpt_inference")

if alerts:
    print("🚨 Alerts detected:")
    for alert in alerts:
        print(f"  {alert}")
        # Send to your alerting system (Slack, email, PagerDuty, etc.)
else:
    print("✓ All systems healthy")
```