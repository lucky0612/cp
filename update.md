apiVersion: monitoring.cnrm.cloud.google.com/v1beta1
kind: MonitoringAlertPolicy
metadata:
  name: alert-coml-5913-no-response-usc1
  labels:
    component_id: coml-5913
    contact_name: sre
    project_id: prj-dv-sphere-0721
spec:
  projectRef:
    external: "prj-dv-sphere-0721"
  displayName: "No Response Alert: Response Rate Drop (coml-5913 /rest/* usc1)"
  enabled: true
  combiner: OR
  conditions:
    - displayName: "Response rate dropped significantly for 5 minutes"
      conditionMonitoringQueryLanguage:
        query: |
          fetch k8s_container
          | metric 'workload.googleapis.com/http.server.request.count'
          | filter (
              metric.app_id == '2852' &&
              metric.component_id == 'coml-5913' &&
              metric.namespace == 'app-2852-default' &&
              metric.http_route =~ "/rest/.*" &&
              resource.location == 'us-central1'
          )
          | filter metric.http_response_status_code != ''
          | group_by [], [value_request_count_aggregate: sum(val())]
          | {
              align rate(5m)
              | value [current_rate_5m: val()]
              align rate(1h) 
              | value [avg_rate_1h: val()]
          }
          | join
          | filter avg_rate_1h > cast_units(0.001, "1/s")
          | value current_rate_5m / avg_rate_1h
          | condition val() < 0.90
        duration: "300s"
        trigger:
          count: 1

  alertStrategy:
    notificationRateLimit:
      period: "300s"
    autoClose: "1800s"

  notificationChannels:
    - external: "projects/prj-dv-sphere-0721/notificationChannels/sphere-notification-channel"
    - external: "projects/prj-dv-sphere-0721/notificationChannels/refsre-notification-channel"

  documentation:
    content: |
      ## No Response Alert Detected!
      
      **Service:** coml-5913
      **Location:** us-central1
      **Routes:** /rest/*
      
      Response rate has dropped below 90% of baseline for 5+ minutes.
      This indicates requests may be timing out without responses.
      
      **Actions:**
      1. Check pod status: kubectl get pods -l component=coml-5913 -n app-2852-default
      2. Check logs: kubectl logs -l component=coml-5913 -n app-2852-default --tail=50
      3. Verify load balancer health
      4. Check resource utilization
    mimeType: "text/markdown"
