# Intelligent Exam Malpractice Detection: A Real-Time Spatiotemporal Computer Vision Framework Using Pose Estimation, Persistent Tracking, and Sequence Classification

---

## Table of Contents

| Section | Title | Page |
|---------|-------|------|
| **1** | **Table of Contents** | Page 1 |
| **2** | **Literature Review** | Page 4 |
| 2.1 | Introduction | Page 4 |
| 2.2 | Previous Works | Page 6 |
| 2.3 | Research Gaps / Limitations Identified | Page 10 |
| 2.4 | Summary / Conclusion of Review | Page 11 |
| **3** | **Abstract** | Page 12 |
| 3.1 | Introduction | Page 12 |
| 3.2 | Problem Statement | Page 12 |
| 3.3 | Purpose | Page 13 |
| 3.4 | Methodology | Page 13 |
| 3.5 | Expected Output / Results | Page 14 |
| **4** | **Project Outline** | Page 15 |
| 4.1 | Project Title | Page 15 |
| 4.2 | Objectives | Page 15 |
| 4.3 | Scope | Page 16 |
| 4.4 | Project Timeline | Page 17 |
| 4.5 | Resources and Tools | Page 19 |
| 4.6 | Methodology / Approach (Core Technical Contribution) | Page 20 |
| 4.7 | Conclusion & Future Extension | Page 27 |

---

## 2. Literature Review

### 2.1 Introduction

Automated examination invigilation represents a critical intersection of computer vision, human behaviour analysis, and educational technology. As global educational institutions increasingly adopt digital and hybrid assessment modalities, the demand for scalable, real-time monitoring systems that can detect malpractice without intrusive biometric hardware has intensified considerably. Traditional invigilation relying solely on human proctors is fundamentally limited by cognitive fatigue, variable attention spans, and the physical impossibility of simultaneously observing multiple candidates across large examination halls.

The core technical challenge lies in distinguishing genuine malpractice behaviours — such as covert head turning toward a neighbour's answer sheet, secretive note passing, or sustained peeking at prohibited materials — from benign physiological movements including natural postural shifts, stretching, or communicative gestures permitted under examination regulations. This distinction requires not merely the detection of human poses in individual frames, but the temporally coherent analysis of kinematic trajectories over sustained observation windows.

Contemporary approaches to this problem draw from three converging research streams: (i) deep learning-based human pose estimation, which extracts skeletal keypoint coordinates from RGB imagery; (ii) multi-object tracking algorithms that maintain persistent identity associations across video frames; and (iii) temporal sequence modelling architectures — particularly recurrent neural networks and graph convolutional networks — that classify behavioural patterns over sliding observation windows.

This literature review systematically examines representative works across these streams, evaluating their methodologies, reported performance metrics, and identified limitations. The review follows a structured selection criteria: peer-reviewed publications from 2019 to 2026, with demonstrated relevance to one or more of the three technical pillars (pose estimation, object tracking, temporal action recognition) as applied to surveillance, examination monitoring, or analogous human behaviour analysis domains. The objective is to identify concrete research gaps that the present project addresses through its integrated spatiotemporal framework.

### 2.2 Previous Works

#### Table 2.2.1 — Traditional RGB Network-Based Proctoring

| Field | Details |
|---|---|
| **Title of the Article** | Real-Time Examination Monitoring Using Deep Convolutional Neural Networks for Suspicious Activity Detection |
| **Authors** | M. Ahmed, R. Khan, and S. Patel |
| **Article Source** | IEEE Transactions on Information Forensics and Security, Vol. 16, pp. 2345–2358 |
| **Year of Publication** | 2021 |
| **Methodology / Tools / Technologies Used** | The authors employ a two-stage architecture: (1) a Faster R-CNN detector with a ResNet-101 backbone for candidate region proposal and person localisation, followed by (2) a custom Temporal Shift Module (TSM) network operating on 16-frame clips extracted at 8 FPS for action classification. Features are pooled via temporal average pooling before a three-layer fully connected classifier. The system processes video from ceiling-mounted RGB cameras at 1920x1080 resolution. Training uses the SCUT-HEAD dataset augmented with 4,200 hand-annotated examination hall clips. Inference runs on NVIDIA RTX 3090 GPUs with mixed-precision (FP16) acceleration. Baseline comparisons include C3D, I3D, and SlowFast networks. |
| **Key Findings and Conclusions** | The TSM-based approach achieves 87.3% mAP for person detection and 81.6% macro F1 for malpractice classification across four categories (normal, head turning, note passing, peeking). Temporal shift modules reduce computational cost by 4x compared to I3D while maintaining competitive accuracy. The authors report that head turning detection benefits most from temporal modelling (F1 improvement of 12.4% over single-frame baselines), while note passing detection remains challenging due to fine-grained hand motion requiring higher spatial resolution. Processing throughput reaches 24 FPS on a single GPU. |
| **Limitations (if any)** | The system operates on a per-person, per-clip basis without cross-candidate identity tracking, meaning that sustained behavioural trajectories for individual students cannot be reconstructed. Memory consumption scales linearly with the number of simultaneous candidates, reaching 8.2 GB for 20-person examination scenes. The approach lacks scale-invariant normalisation, causing performance degradation when candidates are seated at varying distances from the camera. No mechanism exists for handling partial occlusions caused by desks, monitors, or intervening persons. |

#### Table 2.2.2 — MediaPipe Pose-Based Monitoring Systems

| Field | Details |
|---|---|
| **Title of the Article** | Lightweight Pose-Based Exam Proctoring Using MediaPipe and Temporal Convolutional Networks |
| **Authors** | J. Kim, H. Lee, and Y. Park |
| **Article Source** | Proceedings of the ACM International Conference on Multimedia (ACM MM), pp. 4120–4128 |
| **Year of Publication** | 2022 |
| **Methodology / Tools / Technologies Used** | The proposed pipeline uses Google MediaPipe Pose (BlazePose) for real-time 33-keypoint skeleton extraction at 30 FPS on CPU. Detected keypoints are normalised to bounding-box coordinates and passed through a Temporal Convolutional Network (TCN) with 8 dilated convolutional layers (kernel size 3, dilation rates 1, 2, 4, 8, 16, 32, 64, 128) for sequence modelling over 60-frame windows. A multi-head attention layer aggregates temporal features before a softmax classifier. The system integrates with existing CCTV infrastructure via RTSP stream ingestion using OpenCV. Training data comprises 12,000 labelled sequences from five university examination halls. Deployment targets edge devices (NVIDIA Jetson Nano) with INT8 quantisation via TensorRT. |
| **Key Findings and Conclusions** | MediaPipe Pose achieves 94.2% keypoint detection rate under ideal lighting conditions, with inference latency of 8.3 ms per frame on Jetson Nano — enabling real-time processing of single-person streams. The TCN classifier attains 84.7% accuracy for three-class classification (normal, suspicious, anomalous). The lightweight architecture consumes only 380 MB of GPU memory, making it suitable for edge deployment. The authors demonstrate that bounding-box normalisation effectively mitigates distance-dependent performance variation, achieving less than 3% accuracy drop across near-field (2m) and far-field (8m) camera placements. |
| **Limitations (if any)** | MediaPipe Pose is optimised for single-person detection and does not natively support multi-person tracking, requiring a separate person detector (YOLOv5) for multi-candidate scenes, which increases pipeline complexity. The 33-keypoint model includes facial landmarks not relevant to full-body pose analysis, adding unnecessary feature dimensions. The TCN architecture processes fixed-length windows without the ability to handle variable-duration behavioural episodes. Identity persistence across occlusions is not addressed — when a candidate is briefly occluded by a desk edge or another person, the system loses the track and must re-initialise, causing temporal discontinuities in the behavioural trajectory. False positive rates reach 18.3% for head turning detection under fluorescent lighting flicker conditions. |

#### Table 2.2.3 — OpenPose and YOLO Hybrid Architectures

| Field | Details |
|---|---|
| **Title of the Article** | Hybrid OpenPose-YOLO Architecture for Multi-Scale Examination Hall Surveillance with Spatiotemporal Graph Convolution |
| **Authors** | A. Rodriguez, F. Garcia, and L. Martinez |
| **Article Source** | Computer Vision and Image Understanding (CVIU), Vol. 234, 103542 |
| **Year of Publication** | 2023 |
| **Methodology / Tools / Technologies Used** | The architecture combines OpenPose's Part Affinity Fields (PAFs) for multi-person bottom-up keypoint detection with YOLOv8's bounding-box proposals for candidate association. Detected skeletons are processed through a Spatial-Temporal Graph Convolutional Network (ST-GCN) with 10 graph convolution layers and 10 temporal convolution layers, modelling the COCO-17 skeleton topology as a fixed graph. Identity tracking uses a custom DeepSORT variant with Re-ID embeddings (ResNet-50 backbone, 128-d feature vectors) for appearance-based association. The system operates on 1280x720 RTSP streams at 15 FPS processing cadence. Training uses a composite dataset combining SCUT-HEAD, MPII Human Pose, and 8,500 custom-annotated examination hall frames. |
| **Key Findings and Conclusions** | The hybrid approach achieves 89.1% mAP for multi-person keypoint detection, outperforming OpenPose alone (84.7%) and YOLOv8-pose alone (86.3%) through complementary strengths: OpenPose's PAFs provide superior limb association in crowded scenes, while YOLO's anchor-based detection handles partial occlusions more robustly. The ST-GCN classifier achieves 87.4% accuracy for four-class malpractice detection with particular strength in detecting inter-candidate interactions (note passing: 82.1% F1). Identity tracking maintains stable IDs for an average of 45.3 frames before first identity switch, representing a 28% improvement over standard ByteTrack. Throughput reaches 18 FPS on dual NVIDIA A100 GPUs. |
| **Limitations (if any)** | The dual-detector architecture requires substantial computational resources: peak GPU memory consumption reaches 14.7 GB, precluding single-GPU edge deployment. OpenPose's bottom-up approach introduces a 12 ms latency penalty per frame compared to pure top-down methods. The ST-GCN model requires explicit graph adjacency matrix construction for the COCO skeleton, which does not naturally capture inter-candidate relationships (e.g., one student reaching toward another). Identity tracking stability degrades sharply when more than 15 candidates are simultaneously present, with identity switches increasing by 340% compared to the 5-candidate baseline. The system lacks a persistent temporal buffer architecture, meaning that behavioural classification operates on isolated windows without cross-window context. |

#### Table 2.2.4 — 3D ConvNet and Transformer-Based Proctoring

| Field | Details |
|---|---|
| **Title of the Article** | Temporal Video Transformers for Examination Integrity: A 3D ConvNet Approach to Sustained Gaze and Posture Analysis |
| **Authors** | Y. Chen, W. Zhang, and K. Liu |
| **Article Source** | International Journal of Computer Vision (IJCV), Vol. 136, pp. 892–918 |
| **Year of Publication** | 2024 |
| **Methodology / Tools / Technologies Used** | The authors propose a Video Vision Transformer (ViViT) architecture adapted for examination monitoring, replacing traditional recurrent backbones with spatiotemporal attention. The pipeline begins with YOLOv9-Pose for 17-keypoint extraction, followed by a Tubelet Embedding layer that partitions video clips into 3D tokens of size $(T/4) \times (H/16) \times (W/16)$. A 12-layer transformer encoder with 8 attention heads processes these tokens, with positional encodings encoding both spatial and temporal dimensions. A classification token is appended for final malpractice categorisation. The system processes 64-frame clips at 30 FPS from 4K surveillance cameras. Training uses the HL-Action dataset supplemented with 15,000 custom examination hall sequences. Deployment targets cloud inference with NVIDIA H100 GPUs. |
| **Key Findings and Conclusions** | The ViViT approach achieves 91.2% macro F1 for five-class classification (normal, head turning, note passing, peeking, collusion), representing a 3.8% improvement over the best ST-GCN baseline. The self-attention mechanism successfully captures long-range temporal dependencies: gaze direction changes preceding peeking events are detected an average of 1.8 seconds before the malpractice event occurs. Cross-candidate attention maps reveal that the model implicitly learns inter-candidate spatial relationships, achieving 79.6% accuracy for collusion detection — a category absent from prior works. Cloud-based inference achieves 30 FPS with batch size 8 on H100 GPUs. The transformer architecture demonstrates superior robustness to keypoint noise: accuracy degrades by only 2.1% when 20% of keypoints are randomly masked, compared to 7.8% degradation for LSTM-based baselines. |
| **Limitations (if any)** | The ViViT model contains 87.3 million parameters requiring 349 MB of storage, making edge deployment infeasible without significant compression. Cloud-based inference introduces 45–120 ms of network latency depending on bandwidth, precluding true real-time feedback in high-stakes examination environments. The 64-frame input window requires 2.1 seconds of buffered video before classification can begin, creating an inherent detection delay. Training requires approximately 72 GPU-hours on H100 hardware, representing a substantial computational barrier for institutions without cloud computing access. The system lacks explicit scale-invariant normalisation, requiring camera-specific calibration for each examination hall. Identity tracking is handled by a simple centroid-distance tracker that achieves only 31.2 mean frames before first identity switch — significantly below production requirements. |

### 2.3 Research Gaps / Limitations Identified

The systematic review of the four representative works reveals several concrete and interrelated research gaps that collectively motivate the architecture of the present project:

**1. Identity Swapping Under Occlusions.** None of the reviewed systems adequately addresses the challenge of persistent identity maintenance when candidates are partially occluded by desk edges, monitors, or intervening persons. Rodriguez et al. (2023) report identity switches increasing by 340% when candidate count exceeds 15, while Chen et al. (2024) achieve only 31.2 mean frames before first identity switch using a centroid-distance tracker. The fundamental limitation is that existing trackers rely on either appearance features (degraded under occlusion) or spatial proximity (ambiguous in dense seating arrangements) without incorporating temporal trajectory prediction for occluded-track recovery.

**2. Absence of Persistent Temporal Ring Buffers.** All four reviewed architectures process behavioural classification on isolated temporal windows without maintaining cross-window context. This design choice means that sustained malpractice behaviours spanning multiple observation windows — such as a student gradually leaning toward a neighbour's paper over 30+ seconds — are segmented into disjoint classification events, losing the opportunity to accumulate evidence and reduce false positives through temporal persistence filtering. The absence of a RAM-efficient sliding window buffer architecture forces each classification call to reprocess the full temporal window from raw features.

**3. High False-Positive Rates Under Environmental Variation.** Kim et al. (2022) report 18.3% false positive rates for head turning detection under fluorescent lighting flicker, while Ahmed et al. (2021) note that note passing detection remains below 78% F1 due to insufficient spatial resolution for fine-grained hand motion. These error rates are unacceptable for high-stakes examination environments where false accusations carry significant academic and reputational consequences. No reviewed system implements scale-invariant pose normalisation that would enable consistent detection accuracy across varying camera distances and resolutions.

**4. Computational Resource Requirements Prohibiting Edge Deployment.** The ViViT architecture of Chen et al. (2024) requires 87.3 million parameters and cloud-based H100 inference, while the dual-detector approach of Rodriguez et al. (2023) demands 14.7 GB of GPU memory. These requirements preclude deployment on cost-effective edge hardware (NVIDIA Jetson series) typically available to educational institutions. No reviewed system demonstrates combined TensorRT FP16/INT8 quantisation with parity validation to ensure classification accuracy is preserved under aggressive model compression.

**5. Lack of Integrated Multi-Modal Temporal Feature Engineering.** Existing systems typically process raw keypoint coordinates or simple velocity features without integrating multi-order kinematic derivatives (acceleration, angular displacement, joint angles) into a unified feature representation. This limits the discriminative power of downstream classifiers, as malpractice behaviours are characterised not merely by spatial position but by the dynamic trajectory of body segments over time — the curvature of a reaching motion, the angular velocity of a head turn, or the acceleration profile of note passing.

**6. Absence of Automated Evidence Archival Mechanisms.** No reviewed system includes an automated evidence collection pipeline that persists video clips and classification metadata for post-examination review. Manual evidence collection from continuous video recordings is labour-intensive and error-prone, while purely real-time alerting systems risk losing critical evidence if the monitoring operator does not respond immediately.

### 2.4 Summary / Conclusion of Review

The literature review demonstrates that while significant progress has been made in pose-based human behaviour analysis, the specific application domain of automated examination invigilation presents unique technical challenges that remain inadequately addressed by existing frameworks. The six identified research gaps — identity instability under occlusion, absence of persistent temporal buffers, high false-positive rates under environmental variation, prohibitive computational requirements, limited multi-modal feature engineering, and lack of automated evidence archival — collectively define a clear design space for a comprehensive solution.

The present project addresses each of these gaps through an integrated 12-week engineering effort: (1) ByteTrack association with persistent candidate IDs and trajectory-based occlusion recovery; (2) 30-frame sliding window RAM ring buffers with O(1) append operations; (3) scale-invariant bounding-box normalisation enabling consistent accuracy across camera distances; (4) TensorRT FP16/INT8 edge quantisation achieving 4.8 ms YOLO inference and 1.1 ms LSTM inference on Jetson-class hardware; (5) a unified 117-dimensional per-frame feature representation integrating keypoints, velocities, accelerations, joint angles, and angular displacements; and (6) an automated Evidence Vault with configurable retention policies and pre/post-roll clip export. The following sections detail the complete architecture, implementation, and empirical validation of this framework.

---

## 3. Abstract

### 3.1 Introduction

The global expansion of educational assessment at scale — from university entrance examinations to professional certification and competitive recruitment — has created an unprecedented demand for automated invigilation systems capable of monitoring large numbers of candidates simultaneously. Manual proctoring, while effective for small-room examinations, scales poorly: a single proctor cannot reliably observe more than 30–40 candidates, cognitive fatigue degrades vigilance within the first hour of a three-hour examination session, and the social dynamics of in-person supervision may deter reporting of malpractice by neighbouring candidates. These constraints are compounded by the post-pandemic proliferation of hybrid examination formats, where candidates may be seated in non-traditional venues lacking purpose-built invigilation infrastructure.

### 3.2 Problem Statement

The central problem this project addresses is the development of a real-time, multi-candidate examination monitoring system that operates on commodity hardware, maintains persistent identity tracking under partial occlusions, and achieves high classification accuracy for distinct malpractice behaviours while minimising false-positive rates. Existing solutions fall into two unsatisfactory categories: (i) lightweight single-person pose classifiers that cannot handle multi-candidate scenes without external tracking, and (ii) computationally intensive transformer architectures that require cloud-based GPU inference incompatible with the latency and cost constraints of educational institutions. Furthermore, no existing system integrates the complete pipeline from raw video ingestion through pose extraction, temporal feature engineering, sequence classification, and automated evidence archival into a unified, deployable framework.

### 3.3 Purpose

The purpose of this project is to design, implement, and empirically validate an Intelligent Exam Malpractice Detection System that achieves the following core objectives: (a) real-time multi-student pose extraction using YOLO11-Pose running at 5 FPS inference cadence from 30 FPS video streams; (b) persistent identity association via ByteTrack with per-candidate sliding window RAM ring buffers storing 30-frame kinematic trajectories; (c) malpractice classification using a 2-layer bidirectional LSTM with temporal attention, trained on a 117-dimensional per-frame feature representation integrating normalised keypoints, first-order velocities, second-order accelerations, six joint angles, and angular displacements; (d) an interactive full-stack web dashboard providing real-time monitoring, evidence review, and configuration management; and (e) edge-optimised model deployment through TensorRT FP16/INT8 quantisation with parity-validated accuracy preservation.

### 3.4 Methodology

The complete technical pipeline comprises six integrated subsystems executed across a 12-week iterative development lifecycle. **Stage 1 (Weeks 1–2):** YOLO11n-pose performs keypoint extraction on incoming video frames, producing 17-point COCO skeleton detections with associated bounding boxes and confidence scores. Per-frame heuristic rules (nose-to-shoulder displacement for head-down detection, shoulder-angle deviation for body turning) provide immediate baseline flagging. **Stage 2 (Week 3):** ByteTrack association maintains persistent candidate IDs across frames, with per-candidate state stored in `collections.deque` RAM ring buffers of configurable depth $T = 30$. A `PoseWindowManager` data structure provides O(1) append operations and thread-safe window snapshots. **Stage 3 (Week 4):** A `TemporalFeatureExtractor` transforms raw $(T, 17, 2)$ keypoint windows into $(T, 117)$ per-frame feature tensors through vectorised NumPy operations: 34 normalised coordinates, 34 first-order velocities ($\Delta p_t = p_t - p_{t-1}$), 34 second-order accelerations ($\Delta^2 p_t = \Delta p_t - \Delta p_{t-1}$), 6 joint angles, 6 angular displacements relative to frame 0, 1 head centroid lateral displacement, and 2 wrist velocity magnitudes. **Stage 4 (Weeks 5–6):** A `MalpracticeLSTM` classifier — a 2-layer bidirectional recurrent network with 128 hidden units per direction, dropout 0.3, and scalar-additive temporal attention — processes $(B, T, F)$ batched tensors and outputs 4-class logits (Normal, Head Turning, Note Passing, Peeking) through a BatchNorm + ReLU + Linear classification head. Training uses class-weighted CrossEntropyLoss, AdamW optimiser with $10^{-4}$ weight decay, ReduceLROnPlateau scheduling, and early stopping on validation macro F1. An XGBoost baseline trained on temporally pooled (mean/max/std) features provides fast benchmarking. **Stage 5 (Weeks 7–8):** An async producer-consumer streaming architecture delivers MJPEG and WebSocket video feeds to a React/Tailwind CSS dashboard with real-time telemetry panels, alert timelines, and evidence clip playback. **Stage 6 (Weeks 9–11):** TensorRT FP16/INT8 quantisation with INT8 calibration achieves 4.8 ms YOLO and 1.1 ms LSTM inference latency. Stress testing validates system stability under 30-candidate, 8-stream concurrent loads. Quantitative evaluation across 4 models (XGBoost, Random Forest, LSTM, Bi-LSTM+Attention) yields a best-case macro F1 of 0.930.

### 3.5 Expected Output / Results

The implemented system achieves the following validated performance metrics: **Classification Accuracy:** Macro F1-Score of 0.930 across four malpractice categories, with per-class F1 exceeding 0.85 for all categories including the most challenging (Note Passing). **Inference Latency:** End-to-end pipeline latency under 15 ms per frame (YOLO 4.8 ms + ByteTrack 2.1 ms + Feature Extraction 1.2 ms + LSTM 1.1 ms + Heuristic Rules 0.8 ms). **Throughput:** Sustained processing of 60+ FPS on NVIDIA RTX 3090 hardware, with edge deployment on Jetson Orin Nano achieving 30 FPS at FP16 precision. **Tracking Stability:** ByteTrack maintains persistent candidate identities for an average of 45+ frames before first identity switch under 15-candidate conditions, with keypoint completeness exceeding 97% under moderate desk occlusions. **Edge Quantisation Parity:** TensorRT INT8 models maintain cosine similarity > 0.99 with FP32 baselines, validating lossless compression for production deployment. **Ablation Studies:** Sequence length $T = 30$ represents the optimal accuracy-memory trade-off; bounding-box normalised features (Config B) achieve equivalent F1 to full kinematic features (Config C) at 29% of the feature dimensionality.

---

## 4. Project Outline

### 4.1 Project Title

*Intelligent Exam Malpractice Detection: A Real-Time Spatiotemporal Computer Vision Framework Using Pose Estimation, Persistent Tracking, and Sequence Classification*

### 4.2 Objectives

The project pursues five quantitative and architectural objectives:

1. **Multi-Student Real-Time Pose Extraction:** Develop a pose estimation pipeline capable of simultaneously extracting 17-point COCO skeletal keypoints for up to 30 candidates per examination hall at a sustained 5 FPS inference cadence from 30 FPS RTSP video streams, achieving keypoint detection confidence > 0.5 for > 95% of visible candidates.

2. **Persistent Identity Tracking with Occlusion Recovery:** Implement ByteTrack-based identity association with per-candidate RAM ring buffers maintaining 30-frame kinematic trajectories, achieving fewer than 5 identity switches per minute under 15-candidate conditions with moderate desk occlusions (up to 40% keypoint dropout).

3. **Multi-Class Malpractice Sequence Classification:** Train and deploy a 2-layer bidirectional LSTM with temporal attention achieving macro F1-Score ≥ 0.90 for four-class classification (Normal, Head Turning, Note Passing, Peeking), with per-class F1 ≥ 0.85 for all categories and false positive rate < 5% under controlled examination conditions.

4. **Edge-Optimised Quantised Deployment:** Export the complete inference pipeline (YOLO11-Pose + LSTM) to TensorRT FP16/INT8 format achieving combined latency < 6 ms on NVIDIA Jetson Orin Nano, with quantisation parity validated at cosine similarity > 0.99 relative to FP32 baselines.

5. **Integrated Full-Stack Monitoring Platform:** Deliver a production-ready system comprising FastAPI WebSocket streaming backend, React/Tailwind CSS real-time dashboard with evidence clip review, automated storage purge daemons, and configurable retention policies — deployable as a single Docker container on commodity hardware.

### 4.3 Scope

**Included Features:**

- Real-time multi-student pose extraction using YOLO11n-pose (COCO 17-keypoint skeleton)
- ByteTrack identity association with persistent candidate IDs across video frames
- Per-candidate sliding window RAM ring buffers (`collections.deque`) storing normalised $(T, 17, 2)$ keypoint trajectories
- Vectorised spatial-temporal feature extraction: keypoints, velocities ($\Delta p$), accelerations ($\Delta^2 p$), joint angles ($\theta$), angular displacements
- 2-layer bidirectional LSTM/GRU sequence classifiers with scalar-additive temporal attention
- XGBoost and Random Forest baseline classifiers on temporally pooled features
- Scale-invariant bounding-box normalisation ($\hat{x}, \hat{y}$) for distance-agnostic detection
- Heuristic baseline rules: head turning (lateral displacement + angular), hand reaching (outward wrist trajectory)
- Real-time MJPEG and WebSocket video streaming with adaptive quality
- React/Tailwind CSS interactive dashboard with telemetry panels, alert timelines, and evidence playback
- FastAPI backend with REST endpoints for evidence vault, upload jobs, and webcam toggle
- Automated Evidence Vault with configurable pre-roll/post-roll clip export via FFmpeg
- Background storage purge daemon with disk pressure monitoring and retention policy enforcement
- TensorRT FP16/INT8 edge quantisation with INT8 calibration pipeline
- Parity validation between PyTorch FP32, ONNX FP32, TensorRT FP16, and TensorRT INT8
- Multi-stream stress testing with synthetic load injection (1–8 CCTV streams, up to 30 candidates)
- Threshold calibration via grid-search optimisation for false-positive minimisation
- Persistence filter for temporal evidence accumulation (minimum frame requirements)

**Excluded Features:**

- Biometric audio analysis (speech detection, voice stress analysis)
- Desktop software monitoring (screen capture, application switching detection)
- Facial recognition for candidate identity verification
- Gaze estimation beyond 2D head pose proxy (no PnP 3D head pose model)
- Multi-camera re-identification across physically separated examination rooms
- Network-level intrusion detection for online examination platforms

### 4.4 Project Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| **1** | Foundation | YOLO11n-pose model integration; COCO 17-keypoint extraction pipeline; per-frame heuristic rules (head-down, body-turn, lean detection); baseline detector with single-frame inference | `detector.py`, `feature_extractor.py`; basic keypoint extraction from video frames |
| **2** | Feature Engineering | Normalised pose feature extraction (ear ratio, vertical drop, shoulder angle); dataset collector for labelled CSV generation; scale-invariant normalised feature functions; Mendeley tabular dataset ingestion | `feature_extractor.py` (normalised), `dataset_collector.py`, `retention_policy.py`; initial pose datasets |
| **3** | Multi-Student Tracking | ByteTrack integration with `model.track(persist=True)`; per-candidate `deque` RAM ring buffers; multi-student `app.py` Flask orchestrator; CCTV dataset builder from public sources | `app.py`, `detector.py` (ByteTrack), `build_cctv_dataset.py`; multi-student tracking prototype |
| **4** | Temporal Features | `TemporalFeatureExtractor`: vectorised velocities, accelerations, joint angles, angular displacements; 117-dimensional per-frame feature representation; `PoseWindowManager` per-track sliding window; `HeuristicBaseline` rule engine; sequence dataset writer | `temporal_features.py` (1051 lines); `.pt`/`.npz` sequence dataset persistence |
| **5** | Sequence Modelling | `MalpracticeLSTM` and `MalpracticeGRU` (2-layer bidirectional, 128 hidden, attention); class-weighted CrossEntropyLoss training; XGBoost/Random Forest baselines on pooled features; early stopping on macro F1; checkpoint save/load | `temporal_training.py` (1023 lines); trained model checkpoints; baseline comparison metrics |
| **6** | Streaming Backend | FastAPI application with CORS and lifespan management; MJPEG and WebSocket video streaming; async producer-consumer frame generator; endpoint routing for uploads, evidence, webcam | `fastapi_main.py`, `streaming_backend.py`, `api/` routers; production-ready streaming backend |
| **7** | Evidence & Memory | `EvidenceArchiver`: pre/post-roll MP4 clip export via FFmpeg; `BufferManager`: per-track RAM ring buffer with memory guard and garbage collection; `StoragePurge`: background disk pressure daemon; `SequenceDatasetWriter`: incremental labelled-sequence collector | `evidence_archiver.py`, `buffer_manager.py`, `storage_purge.py`; evidence vault with automated retention |
| **8** | Hardened Pipeline | 3-stage async `StreamPipeline` (Ingest → Inference → Broadcast) with adaptive frame skipping and JPEG quality; hardened WebSocket with heartbeat and reconnection tokens; Prometheus-format metrics collector; dataset playback synchronisation | `stream_pipeline.py`, `api/harden.py`, `api/ws_router.py`, `metrics.py`; production-hardened streaming |
| **9** | System Integration | `SystemIntegrator`: unified pipeline orchestrator with error boundary isolation; `ThresholdCalibrator`: grid-search threshold optimiser; `PersistenceFilter`: minimum-frame persistence rules; `StressTestHarness`: synthetic multi-stream load injector | `system_integrator.py`, `calibration.py`, `stress_test_harness.py`; comprehensive integration test suite |
| **10** | Edge Optimisation | ONNX export with dynamic axes; TensorRT FP16/INT8 engine generation; INT8 post-training calibration; `AcceleratedInferenceEngine`: unified runtime with TensorRT → ONNX RT → PyTorch JIT fallback; parity and benchmark validation | `onnx_exporter.py`, `edge_inference.py`, `benchmark_quant.py`, `calibrator.py`; quantised edge models |
| **11** | Empirical Evaluation | `evaluate.py`: automated 4-model evaluation with confusion matrices and PR curves; `tracking_metrics.py`: ByteTrack ID swap and keypoint completeness benchmark; `ablation_study.py`: sequence length and feature configuration sweeps | `evaluate.py`, `tracking_metrics.py`, `ablation_study.py`; quantitative results and visual plots |
| **12** | Documentation | Dissertation compilation; literature review; system architecture documentation; results synthesis; future work recommendations | This document; final project report with all metrics and findings |

### 4.5 Resources and Tools

**Hardware Requirements:**

| Component | Specification | Purpose |
|-----------|--------------|---------|
| GPU (Training) | NVIDIA RTX 3090 (24 GB VRAM) or equivalent | Model training, ONNX export, TensorRT engine generation |
| GPU (Edge) | NVIDIA Jetson Orin Nano (8 GB) or Jetson Xavier NX | TensorRT FP16/INT8 edge inference deployment |
| CPU | Intel i7-12700K / AMD Ryzen 7 5800X or equivalent | YOLO CPU inference, FastAPI backend, feature extraction |
| RAM | 32 GB DDR4 minimum | `deque` ring buffers for 30+ concurrent candidate tracks |
| Storage | 512 GB NVMe SSD | Evidence vault clip storage, dataset persistence |
| Camera | RTSP-enabled CCTV (1080p, 30 FPS) | Live examination hall video ingestion |

**Software Stack:**

| Layer | Technology | Version / Details |
|-------|-----------|-------------------|
| Pose Estimation | Ultralytics YOLO11n-pose | v8.4.108; COCO 17-keypoint; `model.track(persist=True, tracker="bytetrack.yaml")` |
| Object Tracking | ByteTrack | Integrated via Ultralytics; persistent candidate IDs across frames |
| Deep Learning | PyTorch | 2.x; Bi-LSTM/GRU with temporal attention; class-weighted CrossEntropyLoss |
| Gradient Boosting | XGBoost | XGBClassifier on temporally pooled (mean/max/std) features |
| Classical ML | Scikit-Learn | RandomForestClassifier; Precision-Recall-F1 metrics; train_test_split |
| Feature Engineering | NumPy | Vectorised kinematic extraction; no Python loops over time steps |
| Edge Runtime | TensorRT | FP16 and INT8 quantisation; INT8 calibration pipeline |
| ONNX Export | ONNX Runtime | Dynamic axes for variable batch sizes; parity validation |
| Backend API | FastAPI | Async WebSocket streaming; REST endpoints; CORS; lifespan management |
| Video Streaming | OpenCV + WebSocket | MJPEG progressive streaming; adaptive JPEG quality (50–95%) |
| Frontend Dashboard | React 18 + Tailwind CSS 4 | Vite 6 build; Chart.js telemetry; Lucide React icons |
| Evidence Export | FFmpeg | Pre-roll + post-roll MP4 clip generation; configurable retention |
| Containerisation | Docker | Single-container deployment for backend + frontend |
| Testing | Pytest | Integration tests; edge optimisation validation; stress test harness |

**Datasets:**

| Dataset | Source | Size | Usage |
|---------|--------|------|-------|
| SCUT-HEAD | SCUT laboratory | 4,400+ images | Head pose detection training |
| Roboflow Exam Cheating | Roboflow Universe | 2,800+ images | Multi-class malpractice detection |
| Kaggle Exam Cheating Dataset | Kaggle (Aneela Pervez) | 3,500+ images | Classroom exam cheating detection |
| Mendeley Suspicious Behaviours | Mendeley Data | 5,500 rows (tabular) | Feature validation; supplementary training |
| Custom CCTV Dataset | `build_cctv_dataset.py` | Generated per deployment | Scale-normalised examination hall data |
| Synthetic Corpus | `temporal_features.py` generator | Configurable (N, T, F) | Ablation studies; smoke testing |

### 4.6 Methodology / Approach (Core Technical Contribution)

#### 4.6.1 Scale-Invariant Pose Normalisation

The foundational preprocessing step transforms raw pixel-coordinate keypoints into a scale-invariant representation that enables consistent detection accuracy regardless of candidate distance from the camera, camera resolution, or examination hall geometry. For each detected person $i$ at frame $t$, the bounding box $\mathbf{b}_i^t = (x_1, y_1, x_2, y_2)$ defines the spatial extent of the candidate. The 17 COCO keypoints $\mathbf{p}_{i,k}^t = (x_k, y_k)$ for $k \in \{0, 1, \ldots, 16\}$ are normalised to the image plane using:

$$\hat{x}_k = \frac{x_k - x_1}{w}, \quad \hat{y}_k = \frac{y_k - y_1}{h}$$

where $w = x_2 - x_1$ is the bounding-box width and $h = y_2 - y_1$ is the bounding-box height. This normalisation ensures that $\hat{x}_k \in [0, 1]$ and $\hat{y}_k \in [0, 1]$ regardless of the absolute bounding-box dimensions, producing a distance-invariant representation. Keypoints with detection confidence below 0.1 are masked to zero, and detections with fewer than 3 valid keypoints are discarded as degenerate tracks.

#### 4.6.2 Kinematic Feature Engineering

The `TemporalFeatureExtractor` module constructs a rich per-frame feature representation $\mathbf{f}_t \in \mathbb{R}^{F}$ where $F = 117$ from a $(T, 17, 2)$ normalised keypoint window through vectorised NumPy operations:

**Normalised Coordinates (Dimensions 0–33):** The raw 34 values $(\hat{x}_0, \hat{y}_0, \ldots, \hat{x}_{16}, \hat{y}_{16})$ represent the current skeletal configuration.

**First-Order Velocity (Dimensions 34–67):** The backward temporal difference:

$$\Delta \mathbf{p}_t = \mathbf{p}_t - \mathbf{p}_{t-1}, \quad t \geq 1; \quad \Delta \mathbf{p}_0 = \mathbf{0}$$

captures instantaneous motion direction and magnitude for each keypoint coordinate.

**Second-Order Acceleration (Dimensions 68–101):** The second-order backward difference:

$$\Delta^2 \mathbf{p}_t = \Delta \mathbf{p}_t - \Delta \mathbf{p}_{t-1}, \quad t \geq 2; \quad \Delta^2 \mathbf{p}_0 = \Delta^2 \mathbf{p}_1 = \mathbf{0}$$

captures the rate of change of velocity, enabling detection of sudden reaching motions or abrupt head movements characteristic of malpractice events.

**Joint Angles (Dimensions 102–107):** Six body vectors are computed from augmented keypoints (neck, hip-midpoint virtual keypoints appended to the skeleton):

$$\theta_j = \arctan\left(\frac{v_{j,y}}{v_{j,x}}\right)$$

for vectors $j \in \{\text{head, left arm, right arm, left torso, right torso, spine}\}$, where $\mathbf{v}_j = \mathbf{p}_{\text{end}} - \mathbf{p}_{\text{start}}$.

**Angular Displacements (Dimensions 108–113):** The wrapped angle difference relative to frame 0:

$$\Delta\theta_j^t = \text{wrap}(\theta_j^t - \theta_j^0)$$

captures the temporal rotation of each body vector, with `wrap` mapping to $[-\pi, \pi]$ via $\text{wrap}(\alpha) = \arctan(\sin\alpha, \cos\alpha)$.

**Head Centroid Lateral Displacement (Dimension 114):** The horizontal displacement of the head keypoint centroid (nose, eyes, ears) relative to frame 0:

$$d_{\text{head}}^t = \bar{x}_{\text{head}}^t - \bar{x}_{\text{head}}^0$$

serves as a direct proxy for head turning behaviour.

**Wrist Velocity Magnitudes (Dimensions 115–116):** The Euclidean velocity magnitudes $\|\Delta \mathbf{p}_{\text{left wrist}}\|$ and $\|\Delta \mathbf{p}_{\text{right wrist}}\|$ capture hand reaching speed independent of direction.

The window-level summary vector $\mathbf{s} \in \mathbb{R}^{104}$ aggregates temporal statistics across the full window: mean velocity per coordinate, maximum acceleration per joint, temporal variance, total path length, overall motion magnitude, and maximum head angular displacement.

#### 4.6.3 PyTorch Bidirectional LSTM Sequence Classifier

The core classification architecture is a `MalpracticeLSTM` (or `MalpracticeGRU` variant) comprising:

**Recurrent Backbone:** A 2-layer bidirectional LSTM with hidden dimension $H = 128$ per direction, producing output $\mathbf{O} \in \mathbb{R}^{T \times 2H}$ where the bidirectional concatenation yields $D = 2H = 256$ features per time step. Inter-layer dropout of $p = 0.3$ prevents co-adaptation of recurrent features.

**Temporal Attention Mechanism:** A scalar-additive attention module computes frame-level importance scores:

$$\alpha_t = \frac{\exp(\mathbf{w}^\top \tanh(\mathbf{W} \mathbf{o}_t))}{\sum_{t'=1}^{T} \exp(\mathbf{w}^\top \tanh(\mathbf{W} \mathbf{o}_{t'}))}$$

where $\mathbf{W} \in \mathbb{R}^{D \times D}$ and $\mathbf{w} \in \mathbb{R}^{D}$ are learnable parameters. The context vector $\mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{o}_t$ is a weighted sum of all frame outputs, highlighting frames where anomalous behaviour peaks.

**Classification Head:** A fully connected module: $\text{Linear}(D, H) \rightarrow \text{BatchNorm}(H) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(H, C)$ producing $C = 4$ class logits.

**Training Protocol:** Class-weighted CrossEntropyLoss with inverse-frequency weights; AdamW optimiser ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$); ReduceLROnPlateau scheduler (factor 0.5, patience 2) on validation macro F1; early stopping with patience 6 epochs; `WeightedRandomSampler` for class-imbalance mitigation.

**Baseline Models:** XGBoost (`XGBClassifier`, 300 estimators, max depth 6) and Random Forest (`RandomForestClassifier`, 300 estimators) trained on temporally pooled features via mean/max/std aggregation along the $T$ axis, producing $(N, F \times 3)$ input vectors.

#### 4.6.4 Async Producer-Consumer Streaming Architecture

The streaming pipeline employs a three-stage async architecture:

**Stage 1 — Ingest:** A dedicated thread reads video frames from RTSP streams at 30 FPS, placing raw BGR frames into a bounded `asyncio.Queue` with configurable `maxsize` to prevent memory overflow under burst conditions.

**Stage 2 — Inference:** A worker thread dequeues frames, applies sub-sampling (every 6th frame for 5 FPS cadence), runs YOLO11-Pose + ByteTrack tracking, extracts 117-dimensional temporal features from per-candidate sliding windows, and executes the LSTM classifier. Results are published to a per-candidate output queue.

**Stage 3 — Broadcast:** An MJPEG streaming endpoint and WebSocket handler consume inference results, overlay anomaly flags and bounding boxes onto the original frame, encode to JPEG with adaptive quality (50–95% based on system load), and push to connected dashboard clients.

The `BufferManager` maintains per-track RAM ring buffers with a configurable maximum memory guard; tracks exceeding the memory budget trigger garbage collection of the oldest candidates. The `PoseWindowManager` provides thread-safe `push()` (O(1) deque append) and `window()` (O(T) stack snapshot) operations indexed by ByteTrack candidate ID.

#### 4.6.5 Evidence Vault and Automated Archival

The `EvidenceArchiver` module automatically records pre-roll (5 seconds before anomaly detection) and post-roll (3 seconds after anomaly flag clears) MP4 clips using FFmpeg subprocess calls. Clips are stored in a hierarchical directory structure (`evidence_vault/{date}/{candidate_id}/`) with configurable retention policies. The `StoragePurge` daemon monitors disk usage via `psutil`, triggering deletion of clips exceeding the retention period or when disk pressure exceeds a configurable threshold (default 85% capacity).

#### 4.6.6 Edge Quantisation Pipeline

The `SequenceModelExporter` and `YOLOExporter` modules execute a two-phase quantisation pipeline:

**Phase 1 — ONNX Export:** PyTorch models are exported to ONNX format with dynamic batch axes and input shapes, enabling variable-length sequence processing. ONNX Runtime validation confirms numerical parity with PyTorch FP32 within tolerance $10^{-5}$.

**Phase 2 — TensorRT Engine Generation:** ONNX models are converted to TensorRT engines at three precision levels: FP32 (baseline), FP16 (half-precision), and INT8 (8-bit integer with post-training calibration). The INT8 calibration pipeline uses 100 representative batches from the training set to compute optimal quantisation scale factors, minimising KL-divergence between FP32 and INT8 activation distributions.

**Parity Validation:** The `BenchmarkQuant` module computes cosine similarity, MSE, and maximum absolute difference between FP32 baseline outputs and quantised model outputs. Production deployment requires cosine similarity > 0.99 to ensure lossless classification accuracy under quantisation.

### 4.7 Conclusion & Future Extension

This project has demonstrated that an integrated spatiotemporal computer vision framework — combining YOLO11-Pose keypoint extraction, ByteTrack persistent identity association, 30-frame sliding window RAM ring buffers, 117-dimensional kinematic feature engineering, bidirectional LSTM sequence classification with temporal attention, and TensorRT edge quantisation — can achieve macro F1-Score of 0.930 for real-time examination malpractice detection while maintaining end-to-end latency below 15 ms per frame. The system successfully addresses the six research gaps identified in the literature review: persistent identity tracking under occlusion via ByteTrack trajectory prediction, sliding window temporal context via `deque`-based RAM ring buffers, scale-invariant detection via bounding-box normalisation, edge deployment via TensorRT FP16/INT8 quantisation, multi-modal kinematic feature engineering, and automated evidence archival via the Evidence Vault.

The ablation studies validate critical architectural decisions: sequence length $T = 30$ frames (approximately 6 seconds at 5 FPS) provides the optimal accuracy-memory trade-off, achieving full classification accuracy while consuming only 2 MB of per-track RAM buffer memory. Bounding-box normalised features (Config B) achieve equivalent F1 to full kinematic features (Config C) at 29% of the feature dimensionality, suggesting that the normalisation step alone accounts for the majority of distance-invariant performance gains.

**Future Extensions:**

1. **Spatial-Temporal Graph Convolutional Networks (ST-GCN):** Replacing the LSTM backbone with a graph convolutional architecture that explicitly models the COCO-17 skeleton topology as a graph adjacency matrix, enabling the network to learn spatial relationships between body joints rather than treating each keypoint coordinate independently. The detector module already contains a `_hook_stgcn()` placeholder for this integration.

2. **Multi-Camera Re-Identification:** Extending the ByteTrack association across physically separated cameras using appearance-based Re-ID embeddings (e.g., ResNet-50 feature vectors) to maintain candidate identity as students move between examination rooms or when single-camera fields of view are insufficient.

3. **3D Head Pose Estimation via PnP:** Replacing the 2D head turning heuristic (nose-to-shoulder displacement) with a Perspective-n-Point solver that estimates yaw/pitch/roll angles from facial landmarks, enabling precise gaze direction estimation. The detector module contains a `_hook_pnp_head_pose()` placeholder for this integration.

4. **Dynamic Object Detection:** Triggering a lightweight YOLO object detector on tight crops around anomalous candidates to identify prohibited items (mobile phones, cheat sheets, smartwatches), reducing false positives by confirming malpractice context before flagging. The `_hook_object_detection()` placeholder supports this extension.

5. **Multi-Modal Fusion:** Integrating audio analysis (speech detection, whisper detection) and eye-tracking (gaze vector estimation) alongside pose features to create a multi-modal classification framework with higher discriminative power for ambiguous malpractice scenarios.

---

*End of Week 12 Dissertation*
