resource "google_logging_metric" "stage2a_fast_exit_count" {
  name        = "${var.name_prefix}_stage2a_fast_exit_count"
  description = "Count of MDM decisions resolved directly at Stage-2A photometric fast-exit."
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.orchestrator.name}\" AND jsonPayload.terminal_state=\"STAGE2A_FAST_EXIT\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "stage1_resolved_count" {
  name        = "${var.name_prefix}_stage1_resolved_count"
  description = "Count of MDM decisions resolved by Stage-1 readout without Stage-2 escalation."
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.orchestrator.name}\" AND jsonPayload.terminal_state=\"STAGE1_RESOLVED\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "stage2_escalation_count" {
  name        = "${var.name_prefix}_stage2_escalation_count"
  description = "Count of MDM decisions escalated to Stage-2 verifier due to hesitation or conformal ambiguity."
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.orchestrator.name}\" AND jsonPayload.escalated=true"

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "abstain_oos_count" {
  name        = "${var.name_prefix}_abstain_oos_count"
  description = "Count of MDM decisions safely abstaining under out-of-scope or insufficient evidence."
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.orchestrator.name}\" AND jsonPayload.terminal_state=\"ABSTAIN_OOS\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "retry_pending_quarantine_count" {
  name        = "${var.name_prefix}_retry_pending_quarantine_count"
  description = "Count of transient upstream timeouts/throttles quarantined as RETRY_PENDING (never false-0)."
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.orchestrator.name}\" AND jsonPayload.terminal_state=\"RETRY_PENDING\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_logging_metric" "contract_violation_count" {
  name        = "${var.name_prefix}_contract_violation_count"
  description = "Count of schema or logprob alignment violations rejected fail-closed."
  filter      = "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${google_cloud_run_v2_service.orchestrator.name}\" AND jsonPayload.terminal_state=\"REJECTED_CONTRACT_VIOLATION\""

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}

resource "google_monitoring_dashboard" "mdm_funnel_overview" {
  dashboard_json = jsonencode({
    displayName = "MDM Calibrated Decision Funnel (${var.name_prefix})"
    gridLayout = {
      columns = "2"
      widgets = [
        {
          title = "Stage-1 Resolved vs. Stage-2 Escalations"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.stage1_resolved_count.name}\""
                  }
                }
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.stage2_escalation_count.name}\""
                  }
                }
              }
            ]
          }
        },
        {
          title = "SLA Quarantine (RETRY_PENDING) & Contract Violations"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.retry_pending_quarantine_count.name}\""
                  }
                }
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.contract_violation_count.name}\""
                  }
                }
              }
            ]
          }
        }
      ]
    }
  })
}
