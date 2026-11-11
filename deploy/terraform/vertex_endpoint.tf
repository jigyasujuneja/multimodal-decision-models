resource "google_vertex_ai_model" "systemone" {
  count        = var.enable_vertex_systemone_endpoint ? 1 : 0
  project      = var.project_id
  display_name = "${var.name_prefix}-systemone-model"
  region       = var.region
  labels       = var.labels

  container_spec {
    image_uri     = var.vertex_serving_container_image
    predict_route = "/v1/systemone"
    health_route  = "/healthz"

    env {
      name  = "MDM_VERTEX_MACHINE_TYPE"
      value = var.vertex_machine_type
    }

    env {
      name  = "MDM_VERTEX_ACCELERATOR"
      value = "${var.vertex_accelerator_type}:${var.vertex_accelerator_count}"
    }

    ports {
      container_port = 8080
    }
  }

  lifecycle {
    precondition {
      condition     = var.vertex_serving_container_image != ""
      error_message = "vertex_serving_container_image must be set to a @sha256 digest when enable_vertex_systemone_endpoint is true."
    }
    precondition {
      condition     = var.vertex_max_replica_count >= var.vertex_min_replica_count
      error_message = "vertex_max_replica_count must be greater than or equal to vertex_min_replica_count."
    }
  }
}

resource "google_vertex_ai_endpoint" "systemone" {
  count        = var.enable_vertex_systemone_endpoint ? 1 : 0
  project      = var.project_id
  name         = var.vertex_endpoint_id
  display_name = "${var.name_prefix}-systemone-endpoint"
  location     = var.region
  labels       = var.labels
}
