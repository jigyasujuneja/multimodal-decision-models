resource "google_artifact_registry_repository" "mdm_images" {
  location      = var.region
  repository_id = "${var.name_prefix}-containers"
  description   = "Container registry for MDM CascadeDecisionOrchestrator and Stage-1 serving images."
  format        = "DOCKER"
  labels        = var.labels

  cleanup_policy_dry_run = false
}

resource "google_storage_bucket" "calibration_artifacts" {
  name                        = "${var.project_id}-${var.name_prefix}-calibration"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  labels                      = var.labels

  versioning {
    enabled = true
  }

  dynamic "encryption" {
    for_each = var.kms_key_name != "" ? [var.kms_key_name] : []
    content {
      default_kms_key_name = encryption.value
    }
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 10
    }
    action {
      type = "Delete"
    }
  }
}
