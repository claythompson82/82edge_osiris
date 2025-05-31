{{/*
Expand the name of the chart.
*/}}
{{- define "osiris.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If DNS name is longer than 63 chars, we'll truncate it and remove trailing dashes.
*/}}
{{- define "osiris.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart version as a label
*/}}
{{- define "osiris.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "osiris.labels" -}}
helm.sh/chart: {{ include "osiris.chart" . }}
{{ include "osiris.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "osiris.selectorLabels" -}}
app.kubernetes.io/name: {{ include "osiris.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "osiris.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "osiris.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the appropriate apiVersion for deployment.
*/}}
{{- define "osiris.deployment.apiVersion" -}}
{{- if .Capabilities.APIVersions.Has "apps/v1" }}
{{- print "apps/v1" }}
{{- else }}
{{- print "extensions/v1beta1" }}
{{- end }}
{{- end -}}
