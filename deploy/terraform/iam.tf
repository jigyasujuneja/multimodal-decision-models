resource "google_service_account" "orchestrator" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-orchestrator-sa"
  display_name = "MDM CascadeDecisionOrchestrator Runtime SA"
  description  = "Least-privilege runtime identity for the Cloud Run MDM orchestrator."
}

resource "google_service_account" "vertex_serving" {
  count        = var.enable_vertex_systemone_endpoint ? 1 : 0
  project      = var.project_id
  account_id   = "${var.name_prefix}-vertex-sa"
  display_name = "MDM Vertex Stage-1 Serving SA"
  description  = "Least-privilege runtime identity for the optional Vertex AI /v1/systemone endpoint."
}

resource "google_storage_bucket_iam_member" "orchestrator_calibration_reader" {
  bucket = google_storage_bucket.calibration_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_storage_bucket_iam_member" "vertex_calibration_reader" {
  count  = var.enable_vertex_systemone_endpoint ? 1 : 0
  bucket = google_storage_bucket.calibration_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.vertex_serving[0].email}"
}

resource "google_project_iam_member" "orchestrator_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_project_iam_member" "orchestrator_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_project_iam_member" "orchestrator_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.orchestrator.email}"
}

resource "google_secret_manager_secret_iam_member" "orchestrator_secret_accessor" {
  count     = var.upstream_api_key_secret_id != "" ? 1 : 0
  project   = var.project_id
  secret_id = var.upstream_api_key_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.orchestrator.email}"
}
