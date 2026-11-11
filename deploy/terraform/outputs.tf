output "orchestrator_service_uri" {
  description = "HTTPS URI of the deployed Cloud Run CascadeDecisionOrchestrator service."
  value       = google_cloud_run_v2_service.orchestrator.uri
}

output "orchestrator_service_account_email" {
  description = "Email of the least-privilege Cloud Run runtime service account."
  value       = google_service_account.orchestrator.email
}

output "calibration_bucket_name" {
  description = "Name of the versioned GCS bucket storing conformal quantiles, temperature scalers, and LTT gates."
  value       = google_storage_bucket.calibration_artifacts.name
}

output "artifact_registry_repository_id" {
  description = "Artifact Registry repository ID hosting digest-pinned MDM container images."
  value       = google_artifact_registry_repository.mdm_images.repository_id
}

output "vertex_systemone_raw_predict_url" {
  description = "Single-route :rawPredict URL for the optional Vertex AI Stage-1 endpoint (empty when disabled)."
  value = (
    var.enable_vertex_systemone_endpoint
    ? "https://${var.region}-aiplatform.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/endpoints/${google_vertex_ai_endpoint.systemone[0].name}:rawPredict"
    : ""
  )
}

output "vertex_dedicated_deployment_spec" {
  description = "Dedicated GPU machine, accelerator, and replica scaling specification for Stage-1 Vertex AI endpoint model deployment."
  value = {
    enabled           = var.enable_vertex_systemone_endpoint
    endpoint_id       = var.enable_vertex_systemone_endpoint ? google_vertex_ai_endpoint.systemone[0].name : ""
    model_name        = var.enable_vertex_systemone_endpoint ? google_vertex_ai_model.systemone[0].name : ""
    service_account   = var.enable_vertex_systemone_endpoint ? google_service_account.vertex_serving[0].email : ""
    machine_type      = var.vertex_machine_type
    accelerator_type  = var.vertex_accelerator_type
    accelerator_count = var.vertex_accelerator_count
    min_replica_count = var.vertex_min_replica_count
    max_replica_count = var.vertex_max_replica_count
  }
}
