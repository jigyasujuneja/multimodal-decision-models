resource "google_vertex_ai_model" "systemone" {
  count        = var.enable_vertex_systemone_endpoint ? 1 : 0
  display_name = "${var.name_prefix}-systemone-model"
  region       = var.region
  labels       = var.labels

  container_spec {
    image_uri     = var.vertex_serving_container_image
    predict_route = "/v1/systemone"
    health_route  = "/healthz"

    ports {
      container_port = 8080
    }
  }

  lifecycle {
    precondition {
      condition     = var.vertex_serving_container_image != ""
      error_message = "vertex_serving_container_image must be set to a @sha256 digest when enable_vertex_systemone_endpoint is true."
    }
  }
}

resource "google_vertex_ai_endpoint" "systemone" {
  count        = var.enable_vertex_systemone_endpoint ? 1 : 0
  name         = "${var.name_prefix}-systemone-ep"
  display_name = "${var.name_prefix}-systemone-endpoint"
  location     = var.region
  labels       = var.labels
}
