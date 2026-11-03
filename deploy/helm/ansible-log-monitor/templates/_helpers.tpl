{{/*
Helpers for addressing the aws-compatible-storage (S4) subchart from the parent chart.

The subchart's own "aws-compatible-storage.fullname" helper cannot be called with the
parent's root context, so these thin helpers resolve the coordinates from
.Values.global.objectStorage, falling back to the chart defaults.
Keep global.objectStorage.host in sync with aws-compatible-storage.fullnameOverride.
*/}}

{{/* Service name / in-cluster DNS name of the object store. */}}
{{- define "alm.objectStorage.fullname" -}}
{{- .Values.global.objectStorage.host | default "aws-compatible-storage" -}}
{{- end }}

{{/* Secret holding AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY. */}}
{{- define "alm.objectStorage.secretName" -}}
{{- .Values.global.objectStorage.secretName | default (printf "%s-credentials" (include "alm.objectStorage.fullname" .)) -}}
{{- end }}

{{/* S3 API port. */}}
{{- define "alm.objectStorage.apiPort" -}}
{{- .Values.global.objectStorage.apiPort | default "7480" -}}
{{- end }}

{{/* Web UI port (also serves the /api readiness endpoint). */}}
{{- define "alm.objectStorage.uiPort" -}}
{{- .Values.global.objectStorage.uiPort | default "5000" -}}
{{- end }}

{{/* Full S3 endpoint URL for in-cluster clients. */}}
{{- define "alm.objectStorage.endpoint" -}}
{{- printf "http://%s:%s" (include "alm.objectStorage.fullname" .) (include "alm.objectStorage.apiPort" .) -}}
{{- end }}
