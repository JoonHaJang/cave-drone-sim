# 센서 모델

이 문서는 시뮬레이터 센서를 **어떻게 모델링했는지**, **왜 그렇게 했는지**, **실제 센서와 얼마나 맞는지를 어떻게 확인했는지**를 정리합니다. 숫자는 모두 `cavesim/config.py`에 있고, 구현은 `cavesim/sensors.py`(LiDAR, IMU, 거리센서, 기압계)와 `cavesim/sim.py`(오도메트리)에 있습니다.

## 0. 기본 원칙

1. **센서 출력 = 참값 + 오차 항들의 합.** 오차 항마다 근거(데이터시트, 논문, 실측 데이터)를 하나씩 붙였습니다.
2. **센서는 기체에 고정돼 있습니다.** 드론이 가속하려고 기울면 LiDAR, IMU, 거리센서도 같이 기웁니다.
3. **효과마다 따로 켜고 끌 수 있습니다.** 프리셋(`--lidar`, `--imu`, `--odom`)으로 이상적인 센서와 현실적인 센서를 비교할 수 있습니다.
4. **두 단계로 검증합니다.**
   - **구현 검증:** 코드가 파라미터대로 동작하는지 봅니다(`python -m cavesim.validate`).
   - **실측 보정:** 파라미터가 실제 센서와 맞는지 봅니다(EuRoC, CTU SubT 데이터셋).

## 1. 외부 시뮬레이터와 비교, 그리고 결정

센서 모듈을 외부 시뮬레이터(Genesis, Gazebo, CARLA, Isaac Sim, HELIOS++, gnss-ins-sim)에서 가져올지 검토했습니다. 결론은 **지금 코드를 유지하고, 외부에 있고 우리에게 없던 효과만 가져온다**입니다.

| 센서 | 외부 후보의 상태 | 결정 |
|---|---|---|
| LiDAR | Genesis, Gazebo는 가우시안 잡음 수준이고 반사 손실 모델이 없습니다. Isaac Sim RTX와 HELIOS++는 충실도가 높지만 RTX GPU가 필요하거나 오프라인 전용이라, 학생용 Windows 환경에서 실시간으로 돌릴 수 없습니다. | 유지. 모션 왜곡(HELIOS++, 실제 Ouster 동작)과 먼지 감쇠(CARLA·HELIOS++의 대기 감쇠)를 추가 |
| IMU | 네 후보 모두 기본 모델(백색잡음 + 바이어스)은 우리와 같은 계열입니다. 축 간 간섭과 양자화(Genesis, Gazebo), 진동(gnss-ins-sim)만 우리에게 없었습니다. | 유지. 진동, 스케일·정렬 오차, 포화, 양자화를 추가 |
| 오도메트리 | 외부에 "센서 모듈"이 따로 없습니다. 실제 LIO(FAST-LIO2 등)는 ROS와 Linux가 필요합니다. | 유지. 기하학적 퇴화 모델을 추가 |
| 거리센서, 기압계 | Gazebo/PX4 플러그인도 잡음 + 드리프트 수준입니다. | 변경 없음 |

외부 시뮬레이터로 바꾸면 의존성과 설치 부담이 늘어납니다. 그런데 충실도를 실제로 결정하는 건 **모델 식과 파라미터**이고, 파라미터의 신뢰도는 실측 데이터와 비교해야 생깁니다. 그래서 검증은 외부 시뮬레이터가 아니라 실측 데이터셋으로 했습니다.

## 2. LiDAR (Ouster OS0-32)

### 모델

빔 하나마다 아래 순서로 계산합니다(`Lidar.scan`).

1. **기하:** PyBullet 레이캐스트로 동굴 메시까지의 정확한 거리 r을 구합니다.
2. **모션 왜곡:** 열(방위각)마다 그 열을 쏜 순간의 드론 위치와 자세에서 빔을 쏩니다.
3. **반사 손실:** 반사가 돌아올 확률은 아래 세 값을 곱한 것입니다.
   - 거리: `full_detect_range`(8 m)까지는 1이고, 그 뒤로 `max_range`(35 m)에서 `min_detect_prob`(0.8)가 될 때까지 선형으로 줄어듭니다.
   - 입사각: |cos θ|가 `grazing_cos`보다 작으면 비례해서 줄어듭니다.
   - 무작위 손실: 1 − `dropout`
4. **먼지:** 드론이 바닥 가까이 있으면 로터 바람에 날린 먼지가 빔을 가로챕니다(아래 참고).
5. **거리 잡음:** 거리 구간별 σ의 가우시안 잡음을 더합니다(`precision`).

### 각 항목의 근거

| 항목 | 값 | 근거 |
|---|---|---|
| 32채널, 수직 ±45°, 512열, 10 Hz, 최소 0.3 m | | Ouster OS0-32 Rev D 데이터시트. DARPA SubT 우승팀 CERBERUS 드론이 쓴 기종 |
| 거리 잡음 σ | 2 cm(1 m 이하), 1 cm(1–10 m), 1.5 cm(10–15 m), 5 cm(15 m 초과) | 데이터시트. 실측(아래)에서 2 m 이내가 0.7 cm로 사양보다 작게 나와 그대로 둠 |
| `dropout` | **1.5%** | CTU 동굴 실측: 3–8 m에서 반사가 빠진 비율 1.2%. 정합 오차를 감안한 상한 |
| `grazing_cos` | **0.03** | CTU 실측: 벽 법선에서 84–90° 벗어난 빔도 97%가 돌아옴. 거친 암벽은 빛을 사방으로 흩어 보내기 때문 |
| 먼 거리 감쇠 | **8 m부터 줄어서 35 m에서 0.8** | CTU 동굴·광산 실측: 약 10 m까지 0.97, 12–35 m에서 0.86–0.91. 처음 추정(20 m부터, 35 m에서 0.3)은 가까운 쪽은 낙관적이고 먼 쪽은 비관적이었음 |
| 먼지 밀도 | **0.015 /m** (바닥에서) | CTU 광산 실측 0.013 /m(하한값). 처음 추정 0.15 /m는 약 12배 진했음 |
| 먼지 높이, 반경 | 1 m, 2 m | CTU 광산 실측: 약 1 m 위로는 먼지 신호가 사라지고, 먼지 반사 거리는 1.5–2 m에 걸쳐 줄어듦 |

### 모션 왜곡: 왜 필요한가

실제 OS0는 한 바퀴(100 ms)를 512열로 나눠 **열마다 차례로** 쏩니다. 드론이 3 m/s로 날면서 1.5 rad/s로 돌면, 한 스캔 안에서 첫 열과 마지막 열 사이에 30 cm, 8.6°가 어긋납니다. 모든 빔을 한순간에 쏘는 이상적인 모델에서는 이 왜곡이 없어서, 에이전트가 실제로는 필요한 **왜곡 보정(deskew)**을 배울 기회가 없습니다.

- **구현:** 물리 스텝(1/240초)마다 드론 위치와 자세를 기록해 둡니다. 각 열은 자기가 쏘인 시각에 가장 가까운 기록으로 쏩니다.
- **에이전트에게 주는 정보:** 실제 Ouster처럼 빔마다 쏜 시각을 `obs.lidar_t`로 줍니다. 이 시각과 `obs.imu`를 쓰면 에이전트가 직접 왜곡을 보정할 수 있습니다.
- **실측으로 확인한 것** (CTU 데이터의 `t` 필드):
  - 열 j는 j × 0.1952 ms(= 100 ms / 512)에 쏘이고, 같은 열의 빔은 동시에 쏩니다. 모델의 가정과 같습니다.
  - 회전은 **위에서 봐서 시계 방향**이고, **기체 뒤쪽(-180°)**에서 시작합니다. 처음에는 반대 방향으로 구현했다가 이 데이터를 보고 고쳤습니다. 요 회전 중에는 회전 방향에 따라 왜곡 모양이 달라지기 때문입니다.

### 먼지: Beer-Lambert 모델

먼지 구름 안에서 빔이 입자에 부딪히는 거리는 비율 ρ인 지수분포를 따릅니다(Beer-Lambert 감쇠). 그 거리가 벽보다 가깝고 구름 반경 안이면, 벽 대신 먼지가 반사합니다. 밀도는 바닥에 가까울수록 진합니다.

ρ(h) = `dust_density` × max(0, 1 − h / `dust_height`), h = 바닥까지 높이

같은 감쇠 개념이 CARLA(`atmosphere_attenuation_rate`)와 HELIOS++에도 있습니다. 구현 검증에서는 바닥에서 먼지에 걸린 빔 비율이 0.024로, 이론값 1 − e^(−ρd) = 0.025와 일치했습니다.

**실측 보정 (CTU uav22_1, 먼지가 날리는 광산 비행):**

- **먼지 점을 골라내는 기준:** 정답 지도에는 먼지가 없습니다. 그래서 아래 세 조건을 모두 만족하는 반사를 먼지로 셌습니다.
  - 드론에서 3 m 안에서 돌아왔다.
  - 지도의 벽보다 0.5 m 이상 앞에서 돌아왔다.
  - 이웃 칸과 거리가 맞지 않는 고립된 점이다. 먼지 입자는 빔마다 무작위로 맞기 때문입니다.
- **바닥값:** 지도의 빈틈과 정합 오차 때문에 먼지가 없어도 이 기준에 걸리는 반사가 있습니다. 먼지 없는 동굴 시퀀스(uav21_3)에서 0.24%가 나왔고, 이 값을 바닥값으로 뺐습니다.
- **처리본 대조는 쓰지 않았습니다:** CTU의 "먼지 걸러낸" 처리본 토픽은 다운샘플링도 함께 걸려 있어서, 먼지 판정 근거로 쓸 수 없었습니다.

| 바닥까지 높이 | 먼지로 보이는 반사 (바닥값을 뺀 값) |
|---|---|
| 0–0.3 m | 0.55% |
| 0.3–0.6 m | 0.45% |
| 0.6–1.0 m | 0.10% |
| 1.0–1.5 m | 0.04% (사실상 없음) |

밀도는 이렇게 계산했습니다. 희박한 먼지라면 검출 비율 ≈ ρ × (빔마다 먼지가 기준에 걸릴 수 있는 구간 길이)입니다. 구간 길이는 실제 빔의 벽까지 거리로 계산했고, 여기서 ρ₀ ≈ 0.013 /m가 나옵니다. 구름 반경을 1.5–3 m로 바꿔도 0.008–0.014 /m 범위에 있습니다.

이 값은 **하한값**입니다. 0.5 m 안쪽은 기체 프레임과 구분할 수 없어서 보지 못했고, 진하게 뭉친 먼지는 "고립 점" 기준에 걸리지 않습니다. 그래서 0.015 /m로 조금 올려 잡았습니다.

### 실측 보정: CTU SubT Finals 데이터셋

- **데이터:** CTU MRS 그룹이 DARPA SubT 결승 코스에서 자율비행 드론(Ouster OS0-128)으로 기록한 데이터입니다. 정답 지도와 정답 궤적이 함께 있습니다.
- **방법:** 정답 궤적으로 스캔 순간의 LiDAR 위치를 구하고, 정답 지도(10 cm 격자)에 같은 빔을 쏴서 "원래 몇 m에서 반사됐어야 하는지"를 계산했습니다. 그 결과를 실제 측정과 비교했습니다. 우리 센서와 같은 32행이 되도록 128행 중 4행마다 1행을 썼습니다.
- **정합 확인:** 측정 거리와 기대 거리가 0.2 m 안으로 맞는 빔이 동굴에서 64–68%, 광산에서 54%입니다. 0.5 m 안으로는 각각 84–85%, 74%입니다. "범위 안에 벽이 있었나"를 판정하기에는 충분하지만, cm 단위 잡음을 재기에는 부족합니다. 그래서 거리 잡음은 스캔 안에서 평평한 조각에 평면을 맞춰 따로 쟀습니다.

분석하면서 바로잡은 함정이 세 가지 있습니다.

1. **기체 가림:** 처음에는 빔의 절반이 반사가 없는 것처럼 보였습니다. 원인은 CTU 드론의 프로펠러와 다리였습니다. 위쪽 행일수록 심했고(+45°에서 반사 비율 0.16), 막힘 없는 칸은 0.97이었습니다. 그래서 전체 반사 비율이 90% 미만인 칸은 기체에 가린 칸으로 보고 뺐습니다.
2. **좌표계:** 클라우드 점들은 `os_lidar`가 아니라 `os_sensor` 좌표계에 있었습니다. 180° 회전을 넣었을 때는 0.2 m 안에 드는 빔이 25%였고, 바로잡은 뒤 68%가 됐습니다.
3. **거리 잡음과 형상:** 2 m보다 먼 곳에서는 평면 조각이 덮는 실제 면적이 커져서, 암벽의 굴곡이 잡음처럼 섞입니다(2–5 m에서 4.7 cm). 센서 잡음은 조각이 작은 2 m 이내 값(0.7 cm)으로 판단했습니다.

| 기대 거리 | 실측: 동굴 uav21_3 | 실측: 광산 uav22_1 | 시뮬레이터 |
|---|---|---|---|
| 1–5 m | 0.96–0.99 | 0.95–0.97 | 0.98 |
| 5–10 m | 0.97–0.99 | 0.95–0.96 | 0.98 |
| 10–20 m | 0.89–0.97 | 0.88–0.95 | 0.90–0.96 |
| 20–35 m | 0.91 | 0.86 | 0.89 (20–25 m) |

입사각은 광산에서 약간 영향이 보였고(정면 0.97, 75° 넘으면 0.93), 동굴에서는 차이가 없었습니다. 그래서 `grazing_cos`는 0.03으로 작게 유지했습니다.

## 3. IMU (ADIS16448)

### 모델

축마다 측정값은 아래 항들의 합입니다(`Imu.sample`). 샘플은 물리 스텝마다 하나씩, 240 Hz입니다.

측정값 = M · 참값 + 켜질 때 생긴 바이어스 + 천천히 변하는 바이어스 + 백색잡음 + 진동 → 포화 → 양자화

| 항 | 모델 | 값 | 근거 |
|---|---|---|---|
| 백색잡음 | 밀도 N, 이산화하면 σ = N / √Δt | 자이로 3.39e-4 rad/s/√Hz, 가속도 4.0e-3 m/s²/√Hz | RotorS의 ADIS16448 값, Kalibr IMU 노이즈 모델 |
| 바이어스 랜덤워크 | Gauss-Markov(상관시간 1000 s / 300 s) | 자이로 3.88e-5, 가속도 6.0e-3 | RotorS |
| 켜질 때 바이어스 | 드론마다 고정된 상수 | 자이로 8.73e-3 rad/s, 가속도 0.196 m/s² | RotorS |
| 스케일·정렬 오차 (M) | I + 대각(스케일) + 비대각(정렬) | 0.3%, 0.1° | 일반적인 MEMS 수준, Genesis IMU의 축 간 간섭 항 |
| 포화 | 측정 범위에서 잘림 | ±1000°/s, ±18 g | ADIS16448 데이터시트 |
| 양자화 | 1 LSB 단위로 반올림 | 0.04°/s, 0.833 mg | ADIS16448 데이터시트. EuRoC 실측 자이로 값이 0.04°/s의 정수배임을 확인 |
| 로터 진동 | 사인파 + 광대역 (아래) | 축별 값 | **EuRoC 실측으로 보정** |

### 로터 진동: 왜 필요한가

데이터시트 잡음은 센서를 책상 위에 정지시켜 놓고 잰 값입니다. 실제 드론에서는 모터와 프로펠러 진동이 그보다 훨씬 큽니다. Kalibr 문서도 저가 MEMS IMU를 실기체에 쓸 때는 잡음 파라미터를 **10배 이상** 키워야 할 수 있다고 권합니다. 진동이 없는 시뮬레이터에서 만든 추정 필터는 실제 드론에서 IMU를 지나치게 믿게 됩니다.

- **사인파:** 로터 불균형 때문에 회전당 한 번 생기는 진동입니다. 주파수는 √추력에 비례합니다(추력 ∝ 회전수²). 호버링에서 108 Hz입니다.
- **광대역:** 넓은 주파수에 퍼진 흔들림입니다.
- **앨리어싱:** 샘플링 주파수의 절반을 넘는 진동은 접혀서 낮은 주파수로 보입니다. 예를 들어 200 Hz 기록에서 108 Hz 진동은 92 Hz로 보입니다. 실제 IMU 데이터에서도 일어나는 현상이고, 추정 필터에 문제를 일으키는 원인 중 하나입니다.

### 실측 보정: EuRoC MAV 데이터셋

- **데이터:** EuRoC V1_01_easy입니다. AscTec Firefly 헥사콥터에 **우리와 같은 ADIS16448**을 달고 200 Hz로 146초 동안 비행하며 기록했습니다. 6 GB 묶음 zip 안에 있는 IMU 파일만 HTTP Range 요청으로 받았습니다.
- **방법:** 드론의 실제 움직임은 대부분 10 Hz 아래에 있습니다. 그래서 **20–95 Hz 대역**에는 센서 잡음과 진동만 남습니다. 이 대역의 RMS를 축별로 비교했습니다. 비행 데이터라 정지 상태를 전제로 하는 Allan 분산은 쓸 수 없습니다.
- **사인파와 광대역의 비율:** 실측 대역 에너지의 40–65%가 좁은 피크에 몰려 있었습니다. 그래서 절반은 사인파, 절반은 광대역으로 나눴습니다(사인 진폭 A = T, 광대역 σ = T·√(2/3), T는 목표 RMS).

| 20–95 Hz RMS (z = 추력/요 축) | x | y | z |
|---|---|---|---|
| 가속도계, 실측 | 0.47 | 0.65 | 0.88 m/s² |
| 가속도계, 보정 후 | 0.56 | 0.56 | 0.88 m/s² |
| 가속도계, 데이터시트만 | 0.049 | 0.049 | 0.049 m/s² |
| 자이로, 실측 | 0.035 | 0.021 | 0.017 rad/s |
| 자이로, 보정 후 | 0.028 | 0.028 | 0.018 rad/s |

- **데이터시트 잡음과 비교:** 실측 진동은 데이터시트 잡음의 10–18배였습니다.
- **수평 두 축:** 실측은 두 축이 서로 다른데, EuRoC 기체 고유의 비대칭으로 보고 평균을 냈습니다.
- **피크 주파수:** 108 Hz는 200 Hz 기록에서 92 Hz로 접혀 보여서, 실측의 가장 큰 피크(91.6 Hz)와 같은 위치에 옵니다.

### 구현 검증: Allan 편차

`datasheet` 프리셋을 600초 동안 정지 비행시키면서 Allan 편차를 계산하고, 이론값 √(N²/τ + K²τ/3)와 비교했습니다. τ = 0.01–10 s의 8개 값 중 7개가 10% 안에서 맞았습니다. 나머지 하나(가속도계, τ = 10 s)는 기록 하나에서 τ = 10 s 구간이 몇십 개뿐이라 생긴 흔들림입니다. `flight` 프리셋은 짧은 τ에서 진동 때문에 편차가 의도대로 커집니다.

### 프리셋

- `--imu flight` (기본값): 위 모든 항을 켭니다.
- `--imu datasheet`: 백색잡음과 바이어스만 켭니다. 기존 동작과 같습니다.

## 4. 오도메트리 (LIO 출력 흉내)

### 모델

실제 드론의 LiDAR-관성 오도메트리(LIO) 출력을 흉내 냅니다. 오차는 아래 항들로 구성됩니다.

- **이동 거리 비례 오차:** 위치 잡음 σ = `pos_noise_per_m` · √(이동 거리)
- **드론별 스케일 오차:** 드론마다 고정된 값입니다.
- **요 랜덤워크:** 시간이 지나면서 쌓이는 방향 오차, 그리고 회전한 각도에 비례하는 방향 오차입니다.
- **기하학적 퇴화 (새로 추가):** 아래 참고

### 기하학적 퇴화: 왜 필요한가

LIO는 스캔과 지도를 맞춰서 위치를 구합니다. 그런데 **어떤 벽도 마주하지 않는 방향**, 예를 들어 매끈하고 곧은 터널의 축 방향으로는 스캔을 맞춰도 위치가 정해지지 않습니다. DARPA SubT 팀들이 실제로 겪은 대표적인 실패가 이것입니다. 거리에만 비례하는 오차 모델로는 이런 실패를 재현할 수 없습니다.

Zhang et al.(ICRA 2016)의 방법을 따릅니다.

1. 매 스캔에서 LiDAR가 맞힌 벽 법선 n으로 행렬 H = 평균(n nᵀ)을 만듭니다.
2. H의 고유값 λ가 `degen_lambda`(0.05)보다 작은 고유벡터 방향을 퇴화 방향으로 봅니다. 퇴화 정도는 w = 1 − λ / `degen_lambda`(0에서 1 사이)입니다.
3. 그 방향으로만 거리 비례 잡음을 최대 `degen_gain`(8배)까지 키우고, 제자리 비행 중에도 시간에 따라 위치가 흘러가게 합니다(`degen_drift`).

테스트(`tests/test_sensors.py`)로 두 가지를 확인합니다.
- 터널 모양의 법선에서 축 방향만 퇴화로 잡힌다.
- 그 터널 안에서 제자리 비행만 해도 축 방향으로 드리프트가 생긴다.

## 5. 거리센서, 기압계

- **하방 거리센서:** Benewake TFmini Plus 데이터시트를 따릅니다. 0.1–12 m, 5 m까지 3 cm, 그 너머는 거리의 0.6%입니다.
- **기압계:** BMP388급입니다. 잡음 0.1 m에 랜덤워크 드리프트 0.02 m/√s가 더해집니다.

외부 시뮬레이터(Gazebo, PX4)의 모델도 같은 수준이라 변경하지 않았습니다.

## 6. 재현 방법

```
conda activate cave-drone
python -m cavesim.validate                                   # 구현 검증: IMU Allan 편차, LiDAR 거리별 통계, 먼지
python -m cavesim.validate --euroc data/euroc/V1_01_easy     # IMU 진동: EuRoC 실측과 비교
python -m cavesim.validate --ctu data/ctu_subt/uav21_3        # LiDAR: CTU 실측과 정답 지도 비교 (pip install rosbags 필요)
pytest                                                       # 효과별 테스트
```

`data/`는 git에 올리지 않습니다(`.gitignore`). 보정이 끝난 뒤 내려받은 데이터는 지웠습니다. 보정 결과는 `config.py`의 값과 주석, 그리고 이 문서의 표에 남아 있습니다. 검증을 다시 돌리려면 아래처럼 다시 받으세요. 명령은 Git Bash 기준이고, 데이터 사용 조건은 각 출처를 확인하세요.

### EuRoC MAV (IMU 진동 보정)

- **출처:** M. Burri et al., *The EuRoC micro aerial vehicle datasets*, IJRR 2016. ETH Research Collection, DOI [10.3929/ethz-b-000690084](https://doi.org/10.3929/ethz-b-000690084)
- **쓰는 파일:** `vicon_room1.zip`(6.0 GB) 안에 있는 `vicon_room1/V1_01_easy/V1_01_easy.zip`의 `mav0/imu0/data.csv`와 `sensor.yaml`

```
mkdir -p data/euroc/V1_01_easy
curl -L -o vicon_room1.zip "https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content"
python -c "import io, zipfile; z = zipfile.ZipFile('vicon_room1.zip'); inner = zipfile.ZipFile(io.BytesIO(z.read('vicon_room1/V1_01_easy/V1_01_easy.zip'))); [open('data/euroc/V1_01_easy/' + n.split('/')[-1], 'wb').write(inner.read(n)) for n in inner.namelist() if 'imu0/' in n and not n.endswith('/')]"
rm vicon_room1.zip
```

서버가 HTTP Range 요청을 지원합니다. 그래서 6 GB 전체를 받지 않고, 안쪽 zip 구간(약 1.1 GB)만 받아서 IMU 파일을 꺼낼 수도 있습니다. 이번 보정에서는 이 방법을 썼습니다. 다만 서버가 요청이 많으면 429(Too Many Requests)로 잠시 막으므로, 작은 요청을 여러 번 보내지 말고 구간 하나를 한 번에 받아야 합니다.

### CTU SubT Finals (LiDAR 반사 비율, 먼지 보정)

- **출처:** CTU Multi-Robot Systems 그룹, [github.com/ctu-mrs/slam_datasets](https://github.com/ctu-mrs/slam_datasets)의 `subt_finals/`. 기록 기체는 P. Petráček et al., RA-L 2021에 설명돼 있습니다.
- **쓰는 시퀀스:** 정답 지도(15 MB), uav21_3(동굴, 먼지 없음, 1.3 GB), uav22_1(광산, 먼지 있음, 7.9 GB). 압축을 풀면 rosbag이 각각 3.5 GB, 20 GB입니다.
- **정답 파일:** 정답 궤적(`trajectory_groundtruth.txt`)과 지도 원점 변환(`fcu_in_map.mat`)은 GitHub 레포에 있습니다.

```
mkdir -p data/ctu_subt/map data/ctu_subt/uav21_3 data/ctu_subt/uav22_1
curl -L "https://nasmrs.fel.cvut.cz/index.php/s/mf0K9r8rmBcAiav/download" | tar -xz -C data/ctu_subt/map
curl -L "https://nasmrs.fel.cvut.cz/index.php/s/bR5O4Z71QNeOsks/download" | tar -xz -C data/ctu_subt/uav21_3
curl -L "https://nasmrs.fel.cvut.cz/index.php/s/RaFXsset6szhWeR/download" | tar -xz -C data/ctu_subt/uav22_1
for s in uav21_3 uav22_1; do for f in trajectory_groundtruth.txt fcu_in_map.mat; do
  curl -L -o data/ctu_subt/$s/$f "https://raw.githubusercontent.com/ctu-mrs/slam_datasets/master/subt_finals/$s/$f"
done; done
pip install rosbags
python -m cavesim.validate --ctu data/ctu_subt/uav21_3
python -m cavesim.validate --ctu data/ctu_subt/uav22_1
```

다른 시퀀스의 주소는 레포의 `subt_finals/download.sh`에 있습니다.

## 7. 남은 한계

- **IMU 진동:** 기체가 다른 EuRoC 헥사콥터 한 시퀀스(146초)로 보정했습니다. 진동의 크기 수준만 믿을 수 있습니다. 실제 ADIS16448은 내부에서 걸러서 내보내는데, 시뮬레이터에는 이 필터가 없습니다.
- **LiDAR 모션 왜곡:** 가장 가까운 물리 스텝의 자세를 씁니다(1/240초 단위). 스텝 사이는 보간하지 않습니다. 또 스캔 주기가 에이전트 주기(10 Hz)와 같다고 가정합니다.
- **LiDAR 먼 거리:** 실측은 12 m 뒤로 거의 평평한 계단 모양인데, 모델은 직선으로 줄어듭니다. 구간별 차이는 6%포인트 이내입니다.
- **먼지:** 밀도는 하한값을 바탕으로 정했습니다. 먼지는 드론 높이로만 정하고, 비행 속도나 바닥 재질(흙, 암반)의 영향은 넣지 않았습니다. 먼지 구름이 생기고 가라앉는 시간 지연도 없습니다.
- **오도메트리 퇴화:** `degen_lambda`, `degen_gain`, `degen_drift`는 추정값입니다. 실제 LIO를 돌려서 비교해야 보정할 수 있습니다.
- **반사도:** 재질별 반사율과 반사 세기(intensity)는 모델링하지 않았습니다.

## 참고 자료

- Ouster OS0 데이터시트 (Rev D): https://ouster.com/downloads
- Kalibr IMU 노이즈 모델: https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
- RotorS 시뮬레이터: https://github.com/ethz-asl/rotors_simulator
- O. Woodman, *An introduction to inertial navigation*, Cambridge UCAM-CL-TR-696: https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-696.pdf
- J. Zhang, M. Kaess, S. Singh, *On degeneracy of optimization-based state estimation problems*, ICRA 2016
- M. Burri et al., *The EuRoC micro aerial vehicle datasets*, IJRR 2016: https://doi.org/10.3929/ethz-b-000690084
- CTU MRS SLAM datasets (SubT Finals): https://github.com/ctu-mrs/slam_datasets
- P. Petráček et al., *Large-Scale Exploration of Cave Environments by Unmanned Aerial Vehicles*, RA-L 2021
- HELIOS++: https://arxiv.org/abs/2101.09154
- CARLA LiDAR: https://carla.org/Doxygen/html/de/d6f/structFLidarDescription.html
- gnss-ins-sim: https://github.com/Aceinna/gnss-ins-sim
- Genesis 센서: https://genesis-world.readthedocs.io/en/latest/api_reference/sensor/imu.html
