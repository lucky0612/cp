# TRAC Helm Chart

## Chart.yaml
```yaml
apiVersion: v2
name: trac
description: A Helm chart for TRAC application deployment
type: application
version: 0.1.0
appVersion: "1.0.0"
keywords:
  - trac
  - application
maintainers:
  - name: "TRAC Team"
```

## values.yaml
```yaml
# Default values for TRAC application
replicaCount: 3

image:
  repository: us-docker.pkg.dev/prj-dv-artifact-registry-5446/rep-dv-us-internal/2852/coml-5912
  tag: c1615/31141b5200c049d4c23915a55bd3672
  pullPolicy: Always

nameOverride: ""
fullnameOverride: ""

# Application configuration
app:
  name: coml-5912
  id: "2852"
  component: coml-5912
  instance: default
  partOf: "2852"
  version: "NA"
  environment: dv
  department: referential_data_services

# Service configuration
service:
  type: ClusterIP
  ports:
    tcp:
      port: 9091
      protocol: TCP
      targetPort: 9091
    httpJmx:
      port: 443
      protocol: TCP
      targetPort: 8080

# Headless service for StatefulSet
headlessService:
  enabled: true
  ports:
    tcp:
      port: 9091
      protocol: TCP
      targetPort: 9091
    httpJmx:
      port: 443
      protocol: TCP
      targetPort: 8080

# ServiceAccount
serviceAccount:
  create: true
  annotations:
    config.kubernetes.io/origin: |
      path: serviceaccount.yaml
    iam.gke.io/gcp-service-account: sv-2852-gsa-coml-5912@prj-dv-sphere-0721.iam.gserviceaccount.com

# Pod Security Context
podSecurityContext:
  allowPrivilegeEscalation: false
  fsGroup: 1000
  runAsGroup: 1000
  runAsNonRoot: true
  runAsUser: 1000

# Security Context
securityContext:
  fsGroup: 1000
  runAsGroup: 1000
  runAsNonRoot: true
  runAsUser: 1000

# Container resources
resources:
  limits:
    cpu: "7"
    memory: 60Gi
  requests:
    cpu: "5"
    memory: 27Gi

# Persistent Volume
persistence:
  enabled: true
  storageClass: cmek-standard-rwo
  accessMode: ReadWriteOnce
  size: 10Gi

# Topology Spread Constraints
topologySpreadConstraints:
  enabled: true
  maxSkew: 1
  topologyKey: topology.kubernetes.io/zone
  whenUnsatisfiable: ScheduleAnyway

# Volume configurations
volumes:
  configMap:
    defaultMode: 420
    items:
      - key: oscardb.properties
        path: oscardb.properties
      - key: sas_logger.json
        path: sas_logger.json
      - key: logback-spring.xml
        path: logback-spring.xml
      - key: log4j2.xml
        path: log4j2.xml

# Vault configuration for secrets
vault:
  enabled: true
  annotations:
    vault.hashicorp.com/auth-config-role: coml-5912
    vault.hashicorp.com/auth-config-service_account: sv-2852-gsa-coml-5912@prj-dv-sphere-0721.iam.gserviceaccount.com
    vault.hashicorp.com/auth-config-type: iam
    vault.security.banzaicloud.io/vault-role: coml-5912
    vault.security.banzaicloud.io/vault-skip-verify: "true"
  mutate: "true"

# Environment variables
env:
  podInstanceName:
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  podPort:
    valueFrom:
      configMapKeyRef:
        key: trac_app_port
        name: trac-coml-5912-config
  podNamespace:
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
  javaOpts: >-
    -Xms8G -Xmx5SG -Dsun.misc.URLClassPath.disableJarChecking=true -Dserver.port=8012 -Dapp.name=trac -Dorg.springframework.boot.logging.LoggingSystem=none
    -Djavax.xml.parsers.SAXParserFactory=com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl --add-opens
    java.base/java.text=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED
    --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.io=ALL-UNNAMED
    --add-opens java.base/java.xml/com.sun.org.apache.xerces.internal.jaxp.datatype=ALL-UNNAMED

# Database passwords (referenced from Vault)
database:
  oscar:
    password: "vault:kv/data/2852/coml-5912/DV04CPRPO#DV04CPRPO#password"
  pplus:
    password: "vault:kv/data/2852/coml-5912/OF44CPRPO#OF44CPRPO#password"
  copper:
    password: "vault:kv/data/2852/coml-5912/DV54CCRDO#DV54CCRDO#password"
  cma:
    password: "vault:kv/data/2852/coml-5912/QF15CPRPO#QF15CPRPO#password"
  cdb:
    password: "vault:kv/data/2852/coml-5912/QA13CEDRU#QA13CEDRU#password"

# TRAC specific configuration
trac:
  appClass: com.cme.sphere.trac.SphereTracApp
  appGroup: msgprocessor
  appPort: "9091"
  containerClass: file.com.cme.sphere.trac.TracContainer
  dataPath: /data
  electionTimeout: "1000"
  enableCli: "true"
  enableDirtyRead: "true"
  headlessServiceName: coml-5912-headless-b
  jmxEnable: "true"
  jmxWebPort: "8080"
  logDir: log
  metadataDir: raft_meta
  peerIds: trac-coml-5912-b-0.coml-5912-headless-b.app-2852-default.9091,trac-coml-5912-b-1.coml-5912-headless-b.app-2852-default.9091,trac-coml-5912-b-2.coml-5912-headless-app-2852-default.9091
  poolerBatchingIntervalMills: "10"
  poolerMaxQueueSize: "100"
  purgeDataFolderOnStart: "true"
  snapshotDir: snapshot
  snapshotIntervalSecs: "30"

# Istio configuration
istio:
  enabled: true
  destinationRule:
    trafficPolicy:
      loadBalancer:
        simple: LEAST_CONN
      tls:
        mode: ISTIO_MUTUAL
  virtualService:
    gateways:
      - asm-system/asm-ingressgateway
    hosts:
      - coml-5912-b-0.trac-2852-default.ant-use5.prj-dv-anthos-host.dev.gcp.cme.com
    http:
      - name: coml-5912-b-0-vs
        route:
          - destination:
              host: coml-5912-headless-b.app-2852-default.svc.cluster.local
              port:
                number: 443

# ConfigMap for logging configuration
configMap:
  log4j2:
    enabled: true
    data: |
      <?xml version="1.0" encoding="UTF-8"?>
      <Configuration status="WARN">
        <Properties>
          <Property name="app.inventory.id">2852</Property>
          <Property name="app.component.inventory.id">5912</Property>
          <Property name="max.files">15</Property>
          <Property name="CONSOLE_LOG_PATTERN">%d{yyyy-MM-dd HH:mm:ss.SSSZ} %5p APP=2852 COMP=coml-5913 MSGID=[%X{message_id}] %c{1} [%t] - %m%n</Property>
          <Property name="FILE_LOG_PATTERN">%d{yyyy-MM-dd HH:mm:ss.SSSZ} %5p APP=%{app.inventory.id} COMP=%{app.inventory.id}-%{component.inventory.id} %m%n</Property>
        </Properties>
        <Appenders>
          <Console name="Console" target="SYSTEM_OUT" follow="true">
            <PatternLayout pattern="${sys:CONSOLE_LOG_PATTERN}"/>
          </Console>
          <Appenders>
            <Loggers>
              <Logger name="org.apache" level="info"/>
              <Logger name="com.cme" level="info"/>
              <Logger name="com.zaxxer.hikari" level="info"/>
              <Root level="info">
                <AppenderRef ref="Console"/>
              </Root>
            </Loggers>
          </Appenders>
        </Configuration>
```

## templates/_helpers.tpl
```yaml
{{/*
Expand the name of the chart.
*/}}
{{- define "trac.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "trac.fullname" -}}
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
Create chart name and version as used by the chart label.
*/}}
{{- define "trac.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "trac.labels" -}}
helm.sh/chart: {{ include "trac.chart" . }}
{{ include "trac.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: {{ .Values.app.component }}
app.kubernetes.io/instance: {{ .Values.app.instance }}
app.kubernetes.io/name: {{ .Values.app.name }}
app.kubernetes.io/part-of: {{ .Values.app.partOf | quote }}
app.kubernetes.io/version: {{ .Values.app.version }}
app_id: {{ .Values.app.id | quote }}
component_id: {{ .Values.app.component }}
department_name: {{ .Values.app.department }}
environment: {{ .Values.app.environment }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "trac.selectorLabels" -}}
app: trac-cluster
app.kubernetes.io/component: {{ .Values.app.component }}
app.kubernetes.io/instance: {{ .Values.app.instance }}
app.kubernetes.io/name: {{ .Values.app.name }}
app.kubernetes.io/part-of: {{ .Values.app.partOf | quote }}
app.kubernetes.io/version: {{ .Values.app.version }}
app_id: {{ .Values.app.id | quote }}
component_id: {{ .Values.app.component }}
department_name: {{ .Values.app.department }}
environment: {{ .Values.app.environment }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "trac.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- printf "sv-%s-ksa-%s" .Values.app.id .Values.app.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create headless service name
*/}}
{{- define "trac.headlessServiceName" -}}
{{- printf "%s-headless-b" .Values.app.name }}
{{- end }}

{{/*
Create StatefulSet name
*/}}
{{- define "trac.statefulSetName" -}}
{{- printf "trac-%s-b" .Values.app.name }}
{{- end }}

{{/*
Create ConfigMap name
*/}}
{{- define "trac.configMapName" -}}
{{- printf "trac-%s-config" .Values.app.name }}
{{- end }}
```

## templates/serviceaccount.yaml
```yaml
{{- if .Values.serviceAccount.create -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "trac.serviceAccountName" . }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "trac.labels" . | nindent 4 }}
  annotations:
    config.kubernetes.io/origin: |
      path: serviceaccount.yaml
    {{- with .Values.serviceAccount.annotations }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
{{- end }}
```

## templates/configmap.yaml
```yaml
{{- if .Values.configMap.log4j2.enabled }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "trac.configMapName" . }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "trac.labels" . | nindent 4 }}
  annotations:
    config.kubernetes.io/origin: |
      path: trac-config.yaml
data:
  log4j2.xml: |
{{ .Values.configMap.log4j2.data | indent 4 }}
  trac_app_class: {{ .Values.trac.appClass }}
  trac_app_group: {{ .Values.trac.appGroup }}
  trac_app_port: {{ .Values.trac.appPort | quote }}
  trac_container_class: {{ .Values.trac.containerClass }}
  trac_data_path: {{ .Values.trac.dataPath }}
  trac_election_timeout: {{ .Values.trac.electionTimeout | quote }}
  trac_enable_cli: {{ .Values.trac.enableCli | quote }}
  trac_enable_dirty_read: {{ .Values.trac.enableDirtyRead | quote }}
  trac_headless_service_name: {{ include "trac.headlessServiceName" . }}
  trac_jmx_enable: {{ .Values.trac.jmxEnable | quote }}
  trac_jmx_web_port: {{ .Values.trac.jmxWebPort | quote }}
  trac_log_dir: {{ .Values.trac.logDir }}
  trac_metadata_dir: {{ .Values.trac.metadataDir }}
  trac_peer_ids: {{ .Values.trac.peerIds }}
  trac_pooler_batching_interval_mills: {{ .Values.trac.poolerBatchingIntervalMills | quote }}
  trac_pooler_max_queue_size: {{ .Values.trac.poolerMaxQueueSize | quote }}
  trac_purge_data_folder_onstart: {{ .Values.trac.purgeDataFolderOnStart | quote }}
  trac_snapshot_dir: {{ .Values.trac.snapshotDir }}
  trac_snapshot_interval_secs: {{ .Values.trac.snapshotIntervalSecs | quote }}
{{- end }}
```

## templates/statefulset.yaml
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {{ include "trac.statefulSetName" . }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "trac.labels" . | nindent 4 }}
  annotations:
    config.kubernetes.io/origin: |
      path: trac-deployer.yaml
spec:
  persistentVolumeClaimRetentionPolicy:
    whenDeleted: Delete
    whenScaled: Retain
  podManagementPolicy: OrderedReady
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "trac.selectorLabels" . | nindent 6 }}
  serviceName: {{ include "trac.headlessServiceName" . }}
  template:
    metadata:
      labels:
        {{- include "trac.selectorLabels" . | nindent 8 }}
      {{- if .Values.vault.enabled }}
      annotations:
        {{- with .Values.vault.annotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
        vault.security.banzaicloud.io/mutate: {{ .Values.vault.mutate | quote }}
      {{- end }}
    spec:
      containers:
      - args:
        - com.cme.sphere.trac.TracContainer
        - localhost:9091
        command:
        - /entrypoint.sh
        env:
        - name: oscar-db-password
          value: {{ .Values.database.oscar.password }}
        - name: pplus-db-password
          value: {{ .Values.database.pplus.password }}
        - name: copper-db-password
          value: {{ .Values.database.copper.password }}
        - name: cma-db-password
          value: {{ .Values.database.cma.password }}
        - name: cdb-db-password
          value: {{ .Values.database.cdb.password }}
        - name: POD_INSTANCE_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_PORT
          valueFrom:
            configMapKeyRef:
              key: trac_app_port
              name: {{ include "trac.configMapName" . }}
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: JAVA_OPTS
          value: {{ .Values.env.javaOpts | quote }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        name: trac-server
        ports:
        - containerPort: 9091
          name: tcp
        - containerPort: 8080
          name: http-jmx
          protocol: TCP
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        securityContext:
          {{- toYaml .Values.securityContext | nindent 10 }}
        volumeMounts:
        - mountPath: /data
          name: data
        - mountPath: /app/conf/oscardb.properties
          name: {{ include "trac.configMapName" . }}-volume
          subPath: oscardb.properties
        - mountPath: /app/conf/sas_logger.json
          name: {{ include "trac.configMapName" . }}-volume
          subPath: sas_logger.json
        - mountPath: /app/conf/logback-spring.xml
          name: {{ include "trac.configMapName" . }}-volume
          subPath: logback-spring.xml
        - mountPath: /app/conf/log4j2.xml
          name: {{ include "trac.configMapName" . }}-volume
          subPath: log4j2.xml
      nodeSelector:
        cloud.google.com/compute-class: compute-general
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      serviceAccountName: {{ include "trac.serviceAccountName" . }}
      {{- if .Values.topologySpreadConstraints.enabled }}
      topologySpreadConstraints:
      - labelSelector:
          matchLabels:
            {{- include "trac.selectorLabels" . | nindent 12 }}
        maxSkew: {{ .Values.topologySpreadConstraints.maxSkew }}
        topologyKey: {{ .Values.topologySpreadConstraints.topologyKey }}
        whenUnsatisfiable: {{ .Values.topologySpreadConstraints.whenUnsatisfiable }}
      {{- end }}
      volumes:
      - configMap:
          defaultMode: {{ .Values.volumes.configMap.defaultMode }}
          items:
          {{- range .Values.volumes.configMap.items }}
          - key: {{ .key }}
            path: {{ .path }}
          {{- end }}
          name: {{ include "trac.configMapName" . }}
        name: {{ include "trac.configMapName" . }}-volume
  updateStrategy:
    type: RollingUpdate
  {{- if .Values.persistence.enabled }}
  volumeClaimTemplates:
  - metadata:
      labels:
        {{- include "trac.labels" . | nindent 8 }}
      name: data
    spec:
      accessModes:
      - {{ .Values.persistence.accessMode }}
      resources:
        requests:
          storage: {{ .Values.persistence.size }}
      storageClassName: {{ .Values.persistence.storageClass }}
  {{- end }}
```

## templates/service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ printf "%s-b-0" .Values.app.name }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "trac.labels" . | nindent 4 }}
  annotations:
    config.kubernetes.io/origin: |
      path: trac-headless.yaml
spec:
  clusterIP: None
  ports:
  - name: http-jmx
    port: {{ .Values.service.ports.httpJmx.port }}
    protocol: {{ .Values.service.ports.httpJmx.protocol }}
    targetPort: {{ .Values.service.ports.httpJmx.targetPort }}
  - name: tcp
    port: {{ .Values.service.ports.tcp.port }}
    protocol: {{ .Values.service.ports.tcp.protocol }}
    targetPort: {{ .Values.service.ports.tcp.targetPort }}
  selector:
    statefulset.kubernetes.io/pod-name: {{ printf "trac-%s-b-0" .Values.app.name }}
  type: {{ .Values.service.type }}
```

## templates/service-headless.yaml
```yaml
{{- if .Values.headlessService.enabled }}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "trac.headlessServiceName" . }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "trac.labels" . | nindent 4 }}
  annotations:
    config.kubernetes.io/origin: |
      path: trac-headless.yaml
spec:
  clusterIP: None
  ports:
  - name: tcp
    port: {{ .Values.headlessService.ports.tcp.port }}
    protocol: {{ .Values.headlessService.ports.tcp.protocol }}
    targetPort: {{ .Values.headlessService.ports.tcp.targetPort }}
  - name: http-jmx
    port: {{ .Values.headlessService.ports.httpJmx.port }}
    protocol: {{ .Values.headlessService.ports.httpJmx.protocol }}
    targetPort: {{ .Values.headlessService.ports.httpJmx.targetPort }}
  selector:
    {{- include "trac.selectorLabels" . | nindent 4 }}
{{- end }}
```

## templates/destinationrule.yaml
```yaml
{{- if .Values.istio.enabled }}
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: {{ printf "%s-b-0" .Values.app.name }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "trac.labels" . | nindent 4 }}
  annotations:
    config.kubernetes.io/origin: |
      path: trac-destinationrule.yaml
spec:
  host: {{ printf "%s-b-0.%s.svc.cluster.local" .Values.app.name .Release.Namespace }}
  trafficPolicy:
    loadBalancer:
      simple: {{ .Values.istio.destinationRule.trafficPolicy.loadBalancer.simple }}
    tls:
      mode: {{ .Values.istio.destinationRule.trafficPolicy.tls.mode }}
{{- end }}
```

## templates/virtualservice.yaml
```yaml
{{- if .Values.istio.enabled }}
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: {{ printf "%s-b-0" .Values.app.name }}
  namespace: {{ .Release.Namespace }}
  labels:
    {{- include "trac.labels" . | nindent 4 }}
  annotations:
    config.kubernetes.io/origin: |
      path: trac-virtualservice.yaml
spec:
  gateways:
  {{- range .Values.istio.virtualService.gateways }}
  - {{ . }}
  {{- end }}
  hosts:
  {{- range .Values.istio.virtualService.hosts }}
  - {{ . }}
  {{- end }}
  http:
  - name: {{ .Values.istio.virtualService.http.name | default (printf "%s-vs" .Values.app.name) }}
    route:
    - destination:
        host: {{ printf "%s.%s.svc.cluster.local" (include "trac.headlessServiceName" .) .Release.Namespace }}
        port:
          number: {{ .Values.istio.virtualService.http.port | default 443 }}
{{- end }}
```

## templates/NOTES.txt
```txt
1. Get the application URL by running these commands:
{{- if .Values.istio.enabled }}
  The TRAC application is accessible through Istio gateway at:
  {{- range .Values.istio.virtualService.hosts }}
  https://{{ . }}
  {{- end }}
{{- else }}
  export POD_NAME=$(kubectl get pods --namespace {{ .Release.Namespace }} -l "app.kubernetes.io/name={{ include "trac.name" . }},app.kubernetes.io/instance={{ .Release.Name }}" -o jsonpath="{.items[0].metadata.name}")
  export CONTAINER_PORT=$(kubectl get pod --namespace {{ .Release.Namespace }} $POD_NAME -o jsonpath="{.spec.containers[0].ports[0].containerPort}")
  echo "Visit http://127.0.0.1:8080 to use your application"
  kubectl --namespace {{ .Release.Namespace }} port-forward $POD_NAME 8080:$CONTAINER_PORT
{{- end }}

2. To check the status of the StatefulSet:
  kubectl get statefulset {{ include "trac.statefulSetName" . }} -n {{ .Release.Namespace }}

3. To view the pods:
  kubectl get pods -l app.kubernetes.io/name={{ include "trac.name" . }} -n {{ .Release.Namespace }}

4. To check the persistent volumes:
  kubectl get pvc -l app.kubernetes.io/name={{ include "trac.name" . }} -n {{ .Release.Namespace }}

5. To view logs:
  kubectl logs -f {{ include "trac.statefulSetName" . }}-0 -n {{ .Release.Namespace }}
```
