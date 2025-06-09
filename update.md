apiVersion: monitoring.cnrm.cloud.google.com/v1beta1
kind: MonitoringAlertPolicy
metadata:
  name: alert-coml-5913-no-response-usc1 # A unique name for this alert policy
  labels:
    component_id: coml-5913
    contact_name: sre # As per your existing labels
    project_id: prj-dv-sphere-0721 # As per your existing labels
spec:
  displayName: "No Response Alert: >5% Request Timeout Rate (coml-5913 /rest/* usc1)"
  enabled: true
  combiner: OR # Standard for a single condition
  conditions:
    - displayName: "Current timeout rate > 5% for 5 continuous minutes"
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
          # Count total requests across all matching time series
          | group_by [], [value_request_count_aggregate: sum(val())] # Total requests
          | {
              # Calculate request rate over the last 5 minutes
              align rate(5m)
              | value [total_request_rate_5m: val()]
          }
          # Get completed requests (those that got a response)
          | union (
              fetch k8s_container
              | metric 'workload.googleapis.com/http.server.request.duration'
              | filter (
                  metric.app_id == '2852' &&
                  metric.component_id == 'coml-5913' &&
                  metric.namespace == 'app-2852-default' &&
                  metric.http_route =~ "/rest/.*" &&
                  resource.location == 'us-central1'
              )
              | group_by [], [value_completed_count_aggregate: sum(val())] # Completed requests
              | {
                  # Calculate completed request rate over the last 5 minutes
                  align rate(5m)
                  | value [completed_request_rate_5m: val()]
              }
          )
          # Join the total and completed request streams
          | join
          # Calculate the timeout rate percentage
          | value timeout_rate_percent = (1 - (completed_request_rate_5m / total_request_rate_5m)) * 100
          # The condition: alert if timeout rate is greater than 5%
          | condition timeout_rate_percent > 5.0

        duration: "300s" # The condition (timeout rate > 5%) must be true continuously for 5 minutes
        trigger:
          count: 1 # The MQL query results in a single boolean time series for the alert

  alertStrategy:
    notificationRateLimit:
      period: "300s" # Limit notifications to one every 5 minutes for this alert
    autoClose: "1800s" # Automatically close incidents after 30 minutes if the condition clears

  notificationChannels:
    # These should be the fully qualified names or the KRM resource names of your notification channels.
    # Based on your values.yaml and compiled notification channels, these are:
    - name: sphere-notification-channel # KRM resource name
    - name: refsre-notification-channel # KRM resource name
    # If using fully qualified names, they would look like:
    # - "projects/prj-dv-sphere-0721/notificationChannels/YOUR_CHANNEL_ID_1"
    # - "projects/prj-dv-sphere-0721/notificationChannels/YOUR_CHANNEL_ID_2"

  documentation:
    content: |
      ## No Response Alert Detected!

      **Service:** coml-5913
      **Location:** us-central1
      **Affected Routes:** /rest/*
      **App ID:** 2852
      **Namespace:** app-2852-default

      **Alert:** More than 5% of requests are timing out without receiving ANY response and this condition has been sustained for **5 minutes**.

      **Policy Name:** $(policy.display_name)
      **Condition Name:** $(condition.display_name)

      **MQL Condition Met:** 'Timeout rate percentage > 5.0%'

      ---

      ### Potential Causes:
      * Complete service outages or pod crashes
      * Load balancer timeouts or misconfigurations  
      * Network connectivity issues or DNS problems
      * Resource exhaustion (CPU/Memory limits exceeded)
      * Backend database connection pool exhaustion
      * Upstream service dependencies completely failing

      ---

      ### Recommended Actions:
      1. **Verify service health:** Check pod status and recent restarts for coml-5913 (You might want to create a pre-configured link for the specific MQL).
      2. **Check infrastructure:** Examine load balancer health and network connectivity in Google Cloud Console.
      3. **Review recent changes:** Check deployment history, configuration changes, and infrastructure updates.
      4. **Assess resource usage:** Monitor CPU, memory, and connection pool utilization.
      5. **Check dependencies:** Verify upstream services and database availability.

      ---

      *This alert was triggered by the policy: '$(policy.name)' *
    mimeType: "text/markdown"
