resource "google_cloud_run_v2_service" "orchestrator" {
  name                = "${var.name_prefix}-orchestrator"
  location            = var.region
  ingress             = var.cloudrun_ingress
  deletion_protection = false
  labels              = var.labels

  template {
    service_account = google_service_account.orchestrator.email
    timeout         = "180s"

    scaling {
      min_instance_count = var.cloudrun_min_instances
      max_instance_count = var.cloudrun_max_instances
    }

    containers {
      image = var.orchestrator_container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = false
      }

      env {
        name  = "MDM_CALIBRATION_BUCKET"
        value = google_storage_bucket.calibration_artifacts.name
      }

      env {
        name  = "MDM_HESITATION_THRESHOLD"
        value = tostring(var.hesitation_threshold)
      }

      env {
        name  = "MDM_CONFORMAL_ALPHA"
        value = tostring(var.conformal_alpha)
      }

      dynamic "env" {
        for_each = var.enable_vertex_systemone_endpoint ? [google_vertex_ai_endpoint.systemone[0].name] : []
        content {
          name  = "MDM_VERTEX_SYSTEMONE_ENDPOINT"
          value = "https://${var.region}-aiplatform.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/endpoints/${env.value}:rawPredict"
        }
      }

      dynamic "env" {
        for_each = var.upstream_api_key_secret_id != "" ? [var.upstream_api_key_secret_id] : []
        content {
          name = "MDM_UPSTREAM_API_KEY"
          value_source {
            secret_key_ref {
              secret  = env.value
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        initial_delay_seconds = 2
        period_seconds        = 5
        failure_threshold     = 6
      }

      liveness_probe {
        http_get {
          path = "/healthz"
          port = 8080
        }
        period_seconds    = 15
        failure_threshold = 3
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "authorized_invokers" {
  for_each = toset(var.invoker_members)

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.orchestrator.name
  role     = "roles/run.invoker"
  member   = each.value
}
