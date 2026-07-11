# Deploying to local Kubernetes (Minikube)

## 1. Start the cluster

```bash
minikube start --driver=docker --cpus=4 --memory=8192
kubectl get nodes   # expect Ready
```

## 2. Build the image and load it into Minikube

Minikube's cluster doesn't share your host's Docker image cache, and this
deployment intentionally uses no registry (`imagePullPolicy: Never` in
`deployment.yaml`), so the image has to be loaded in explicitly:

```bash
docker build -t heart-disease-api:latest ..   # from repo root: docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
```

## 3. Deploy

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods,svc     # wait for pods to reach Running/Ready
```

## 4. Expose and verify

**Option A — LoadBalancer + tunnel (matches `service.yaml`, primary path):**
`minikube`'s docker driver never populates a LoadBalancer's `EXTERNAL-IP`
unless `minikube tunnel` is running (it edits host routes, so it needs sudo).

```bash
# terminal A (leave running)
minikube tunnel

# terminal B
kubectl get svc heart-disease-api   # EXTERNAL-IP should now be populated
curl http://<EXTERNAL-IP>/health
curl -X POST http://<EXTERNAL-IP>/predict -H "Content-Type: application/json" -d @../sample_input.json
```

**Option B — Ingress (avoids `tunnel`'s sudo requirement):**

```bash
minikube addons enable ingress
kubectl apply -f k8s/ingress.yaml
curl --resolve heart-api.local:80:$(minikube ip) http://heart-api.local/health
```

**Quick sanity check without either (NodePort-style, not a substitute for the
LoadBalancer/Ingress deliverable, just useful while debugging):**

```bash
minikube service heart-disease-api --url
```

## 5. Teardown

```bash
kubectl delete -f k8s/service.yaml -f k8s/deployment.yaml
# and, if used: kubectl delete -f k8s/ingress.yaml
```
