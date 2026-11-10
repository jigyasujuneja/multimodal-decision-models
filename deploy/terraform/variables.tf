variable "project_id" {
  description = "Google Cloud project ID hosting the MDM Cloud Run orchestrator and optional Vertex AI endpoint."
  type        = string

  validation {
    condition     = length(trimspace(var.project_id)) >= 6
    error_message = "project_id must be a non-empty Google Cloud project identifier."
  }
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run, Artifact Registry, GCS, and Vertex AI."
  type        = string
  default     = "us-central1"
}

variable "name_prefix" {
  description = "Lowercase resource prefix applied to service accounts, buckets, Cloud Run, and monitoring assets."
  type        = string
  default     = "mdm"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,23}$", var.name_prefix))
    error_message = "name_prefix must start with a lowercase letter and contain 2..24 lowercase alphanumeric or hyphen characters."
  }
}

variable "orchestrator_container_image" {
  description = "Digest-pinned container image URI for the Cloud Run CascadeDecisionOrchestrator service."
  type        = string

  validation {
    condition     = can(regex("^.+@sha256:[0-9a-f]{64}$", var.orchestrator_container_image))
    error_message = "orchestrator_container_image must be pinned to an immutable @sha256:<64-hex> digest."
  }
}

variable "enable_vertex_systemone_endpoint" {
  description = "Provision a dedicated GPU-backed Vertex AI Model and Endpoint for Stage-1 /v1/systemone serving."
  type        = bool
  default     = false
}

variable "vertex_serving_container_image" {
  description = "Digest-pinned container image URI for the optional Stage-1 Vertex AI serving container."
  type        = string
  default     = ""

  validation {
    condition     = var.vertex_serving_container_image == "" || can(regex("^.+@sha256:[0-9a-f]{64}$", var.vertex_serving_container_image))
    error_message = "vertex_serving_container_image must be empty or pinned to an immutable @sha256:<64-hex> digest."
  }
}

variable "vertex_machine_type" {
  description = "Compute machine type for the optional Stage-1 Vertex AI endpoint."
  type        = string
  default     = "g2-standard-8"
}

variable "vertex_accelerator_type" {
  description = "GPU accelerator type attached to the optional Stage-1 Vertex AI endpoint."
  type        = string
  default     = "NVIDIA_L4"
}

variable "vertex_accelerator_count" {
  description = "Number of GPUs per Stage-1 Vertex AI replica."
  type        = number
  default     = 1
}

variable "vertex_min_replica_count" {
  description = "Minimum replica count for the Stage-1 Vertex AI endpoint."
  type        = number
  default     = 1
}

variable "vertex_max_replica_count" {
  description = "Maximum autoscaled replica count for the Stage-1 Vertex AI endpoint."
  type        = number
  default     = 3
}

variable "cloudrun_min_instances" {
  description = "Minimum warm Cloud Run instances for the CascadeDecisionOrchestrator."
  type        = number
  default     = 1
}

variable "cloudrun_max_instances" {
  description = "Maximum autoscaled Cloud Run instances for the CascadeDecisionOrchestrator."
  type        = number
  default     = 20
}

variable "cloudrun_ingress" {
  description = "Cloud Run ingress policy (defaults to internal + Cloud Load Balancing only)."
  type        = string
  default     = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  validation {
    condition = contains([
      "INGRESS_TRAFFIC_INTERNAL_ONLY",
      "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER",
      "INGRESS_TRAFFIC_ALL",
    ], var.cloudrun_ingress)
    error_message = "cloudrun_ingress must be one of INGRESS_TRAFFIC_INTERNAL_ONLY, INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER, or INGRESS_TRAFFIC_ALL."
  }
}

variable "invoker_members" {
  description = "Explicit IAM principal list granted roles/run.invoker on the Cloud Run orchestrator (allUsers and allAuthenticatedUsers are prohibited)."
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for member in var.invoker_members :
      !contains(["allUsers", "allAuthenticatedUsers"], trimspace(member))
    ])
    error_message = "Public unauthenticated invokers (allUsers, allAuthenticatedUsers) are prohibited on MDM decision services."
  }
}

variable "hesitation_threshold" {
  description = "Default normalized Shannon entropy threshold H_raw triggering Stage-2 cascade escalation."
  type        = number
  default     = 0.16
}

variable "conformal_alpha" {
  description = "Target split-conformal miscoverage rate alpha in (0, 1)."
  type        = number
  default     = 0.10
}

variable "upstream_api_key_secret_id" {
  description = "Optional Secret Manager secret ID holding an upstream Stage-1/Stage-2 API credential (never passed as plaintext Terraform state)."
  type        = string
  default     = ""
}

variable "kms_key_name" {
  description = "Optional Cloud KMS CryptoKey resource name for CMEK encryption on the calibration artifact bucket."
  type        = string
  default     = ""
}

variable "labels" {
  description = "Resource labels applied to buckets, Artifact Registry, Cloud Run, and Vertex AI resources."
  type        = map(string)
  default = {
    managed_by = "terraform"
    workload   = "multimodal-decision-models"
  }
}
