# Cave Drone Sim

GPS가 없는 동굴에서 드론 여러 대로 탐사하고, 목표물을 찾아 보고하고, 입구로 돌아오는 시뮬레이터입니다. DARPA SubT Challenge를 단순하게 줄인 형태입니다.

시뮬레이션은 Python이 합니다. PyBullet으로 충돌과 LiDAR를 계산하고, 드론 운동과 통신, 채점도 Python 쪽에서 처리합니다. 브라우저의 Three.js 뷰어는 Python이 보내 주는 동굴, 드론, LiDAR 점, 궤적, 통신 링크를 그리기만 합니다.

학생이 작성하는 코드는 `cavesim.api.Agent`를 상속한 Python 클래스 하나이고, 드론마다 인스턴스가 하나씩 만들어집니다.

## 1. 설치 (처음 한 번)

Windows 기준입니다. 아래 상자 안의 명령을 순서대로 복사해서 붙여 넣으세요. 20분쯤 걸립니다.

### 1단계. Miniforge 설치

1. https://github.com/conda-forge/miniforge/releases/latest 에서 `Miniforge3-Windows-x86_64.exe`를 받아 실행합니다.
2. 설치 창에서 **Just Me**를 고르고, 나머지는 기본값 그대로 **Next**만 누릅니다. 관리자 권한은 필요 없습니다.
3. 시작 메뉴에서 **Miniforge Prompt**를 엽니다. 이후 명령은 전부 이 창에 입력합니다.

**성공 확인**: Miniforge Prompt에 아래를 입력했을 때 버전 번호가 나오면 됩니다.

```
mamba --version
```

> 왜 pip이 아니라 Miniforge인가요? 시뮬레이터가 쓰는 PyBullet은 PyPI(pip)에 Windows용 설치 파일이 없습니다. conda-forge에는 있어서 Miniforge로 설치합니다.

### 2단계. 코드 받기

**주의: 폴더 경로에 한글이 들어가면 안 됩니다.** 설치 도구(mamba)가 한글 경로를 읽지 못합니다. `바탕 화면`이나 `문서` 아래가 아니라 `C:\work`에 받으세요.

git이 있으면:

```
mkdir C:\work
cd /d C:\work
git clone https://github.com/JoonHaJang/cave-drone-sim.git cave-drone-sim
cd cave-drone-sim
```

git이 없으면 GitHub 저장소 페이지에서 **Code → Download ZIP**으로 받습니다. `C:\work`에 압축을 풀고, 풀린 폴더 이름(`cave-drone-sim-main`)을 `cave-drone-sim`으로 바꾼 뒤 아래를 입력합니다.

```
cd /d C:\work\cave-drone-sim
```

**성공 확인**: `dir`을 입력했을 때 `setup_windows.bat`, `environment.yml`, `cavesim` 폴더가 보이면 됩니다.

### 3단계. 환경 설치

```
setup_windows.bat
```

패키지를 내려받느라 5~10분 걸립니다. 끝나면 마지막에 이렇게 나옵니다.

```
numpy BLAS ok
20 s hover test on small_2 ok (PyBullet collision + LiDAR)

OK. Next: python -m cavesim.run
```

`[error] The folder path contains Korean...`이 나오면 2단계로 돌아가 영문 경로로 옮기세요. 그 밖의 오류는 [8. 문제 해결](#8-문제-해결)을 보세요.

### 4단계. 첫 실행

```
conda activate cave-drone
python -m cavesim.run
```

브라우저에 동굴과 드론 3대가 보이면 설치가 모두 끝난 것입니다. 실행 방법은 [2. 실행하기](#2-실행하기)에 자세히 있습니다.

### 설치 스크립트가 하는 일

`setup_windows.bat`은 아래 세 줄을 실행합니다. 스크립트가 안 되면 직접 입력해도 됩니다.

```
mamba env create -f environment.yml -y
conda activate cave-drone
python -m cavesim.check
```

- `environment.yml`: 실제 설치에 쓰는 패키지 목록입니다(PyBullet 포함).
- `requirements.txt`: 같은 목록에서 PyBullet만 뺀 pip용 파일입니다. 이것만으로는 시뮬레이터가 돌아가지 않으니 설치에는 쓰지 마세요.

## 2. 실행하기

### 2-1. 매번 시작할 때

1. 시작 메뉴에서 **Miniforge Prompt**를 엽니다.
2. 프로젝트 폴더로 가서 환경을 켭니다. 줄 맨 앞이 `(base)`에서 `(cave-drone)`으로 바뀌면 됩니다.

   ```
   cd /d C:\work\cave-drone-sim
   conda activate cave-drone
   ```

3. 시뮬레이터를 실행합니다.

   ```
   python -m cavesim.run
   ```

4. 브라우저가 자동으로 열립니다. 안 열리면 주소창에 http://localhost:8000/viewer/?ws=8765 를 입력합니다.
5. 끝낼 때는 Miniforge Prompt 창에서 `Ctrl+C`를 누릅니다. 임무가 끝나도 서버는 켜진 채로 남아 있으니, 다시 실행하기 전에 꼭 `Ctrl+C`로 끄세요. 안 끄고 또 실행하면 포트가 이미 사용 중이라는 오류가 납니다.

### 2-2. 뷰어 조작

- 마우스 왼쪽 드래그: 회전 / 오른쪽 드래그: 이동 / 휠: 확대·축소
- 아래 버튼: 일시정지, 배속(0.5x~최대), 표시 항목 켜고 끄기
- `동굴` 슬라이더: 동굴 벽 투명도. 낮추면 안쪽 드론이 잘 보입니다.
- 왼쪽 위 드론 이름을 누르고 `선택 드론 따라가기`를 켜면 카메라가 그 드론을 따라갑니다.
- 오른쪽 위 작은 화면은 선택한 드론의 1인칭(FPV) 화면입니다. 기체 앞에 고정된 카메라에서 본 모습이라, 동굴 안은 드론 조명이 닿는 곳만 보입니다. 화면 왼쪽 위에는 속도, 벽까지 거리, 공기 교란 크기가 나오고 오른쪽 위에는 자세와 배터리가 나옵니다. 가운데 점선은 인공 수평선입니다.
- `V` 키나 `FPV 크게` 버튼을 누르면 1인칭 화면이 전체 화면으로 커지고, 한 번 더 누르면 돌아옵니다. 숫자 키 `1`~`9`로 드론을 고를 수 있습니다. 아래 `1인칭(FPV)` 체크박스를 끄면 작은 화면이 사라집니다.
- 임무가 끝나면 화면 가운데에 최종 점수가 뜹니다. 누르면 닫힙니다.

### 2-3. 센서 창

왼쪽 위에서 드론을 고른 뒤 `선택한 드론의 센서 창 열기` 아래 버튼을 누르면, 그 센서만 보여 주는 창이 따로 열립니다. 창마다 위쪽에서 드론을 바꿀 수 있고, 차트 위에 마우스를 올리면 값이 나옵니다.

| 버튼 | 보여 주는 것 |
|---|---|
| LiDAR | 거리 영상(32채널 × 512방위)과 위에서 본 수평 스캔 |
| IMU | 가속도계·자이로 3축, 240 Hz 원시값 |
| 고도 | 기압 고도와 실제 고도, 하방 거리센서 |
| 위치 추정 | 오도메트리 궤적과 실제 궤적, 오차 |
| 상태 | 통신 연결, 자세(추정/실제), 탐지된 목표물 |

점선으로 그려진 "실제" 값은 교수자 확인용 정답이고, 에이전트는 받지 못합니다.

창이 안 열리면 브라우저의 팝업 차단을 `localhost`에 대해 풀어 주세요.

### 2-4. 드론 3D 모델 (선택)

드론을 실제 기체 모양으로 보려면 3D 모델을 따로 받아야 합니다. 라이선스 문제로 저장소에는 넣지 않았습니다. 모델이 없어도 시뮬레이션과 채점은 똑같고, 드론이 색 공으로 그려질 뿐입니다.

1. Sketchfab에서 "Bionic dragonfly four-axis UAV aircraft"(작성자 boxgizmo) 페이지를 찾아 로그인한 뒤 **Download 3D Model → glTF**로 받습니다.
2. 압축을 풀어 프로젝트 폴더 바로 아래에 `bionic_dragonfly_four-axis_uav_aircraft\scene.gltf`가 오게 둡니다. 같은 폴더에 `scene.bin`도 있어야 합니다.
3. 뷰어를 새로고침합니다.

0.5 m짜리 모델은 멀리서는 거의 보이지 않아서, 카메라가 7 m보다 멀어지면 색 공과 방향 원뿔이 같이 그려집니다.

### 2-5. 내 에이전트로 실행

`agents\template.py`를 복사해 팀 파일을 만들고, `--agent`로 지정합니다.

```
copy agents\template.py agents\team1.py
python -m cavesim.run --agent agents.team1
```

코드를 고친 뒤에는 `Ctrl+C`로 끄고 다시 실행합니다. 에이전트에서 오류가 나면 Miniforge Prompt에 오류 내용이 찍히고, 그 드론은 그 틱 동안 제자리에 멈춥니다.

### 2-6. 자주 쓰는 실행 예

```
:: 드론 4대, 위치 오차 크게
python -m cavesim.run --agent agents.team1 --drones 4 --odom hard

:: 다른 맵으로
python -m cavesim.run --map small_2

:: 뷰어 없이 점수만. 여러 번 비교할 때는 lite가 훨씬 빠릅니다
python -m cavesim.run --agent agents.team1 --headless --lidar lite

:: 공기 교란 끄기 (알고리즘 디버깅용) / 더 세게
python -m cavesim.run --agent agents.team1 --aero off
python -m cavesim.run --agent agents.team1 --aero hard

:: 새 맵 만들기 (maps\medium_7 이 생김)
python -m cavesim.cavegen --size medium --seed 7

:: 종유석·석순을 두 배로
python -m cavesim.cavegen --size medium --seed 7 --name medium_7_dense --speleothems 2
```

`--speleothems`는 종유석·석순 밀도 배율입니다. `0`이면 하나도 만들지 않고 `2`면 두 배로 만듭니다. 시드가 같으면 밀도를 바꿔도 통로 모양과 목표물 위치는 그대로입니다.

### 2-7. 옵션 전체

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--agent agents.explorer` | 사용할 에이전트 모듈. `모듈:클래스`도 됩니다 | `agents.explorer` |
| `--map medium_1` | `maps` 아래 폴더 이름 | `medium_1` |
| `--drones 3` | 드론 수 (1~9) | 3 |
| `--odom hard` | 오도메트리 품질: `perfect`, `default`, `hard`, `off`(위치를 주지 않음) | `default` |
| `--lidar lite` | LiDAR 사양: `os0-32`(실제 센서 사양), `lite`(빔 1/4, 약 4배 빠름), `ideal`(잡음 없음) | `os0-32` |
| `--aero hard` | 암벽 근처 공기 교란: `off`, `default`, `hard` ([5장](#5-동굴-지형과-공기)) | `default` |
| `--seed 0` | 센서 잡음 시드. 같은 시드면 같은 결과가 나옵니다 | 0 |
| `--speed 5` | 실시간 대비 배속, 0이면 최대 속도 | 1 |
| `--headless` | 뷰어 없이 점수만 출력 | |
| `--wait` | 뷰어가 접속한 뒤에 시작 | |
| `--no-browser` | 브라우저를 자동으로 열지 않음 | |
| `--http-port`, `--ws-port` | 뷰어 포트 | 8000, 8765 |

맵 크기는 `small`, `medium`, `large` 세 가지입니다.

### 2-8. 화면에 보이는 것

- 드론 모델 또는 색 공: 드론의 실제 위치. 색 공 앞쪽 원뿔이 머리 방향입니다.
- 밝은 원뿔: 종유석, 석순, 석주입니다.
- 같은 색 와이어 공과 점선: 드론이 *추정한* 위치와 궤적입니다. 실제 위치와 벌어지는 만큼이 오도메트리 오차입니다.
- 초록 선: 통신 링크(드론끼리, 드론과 기지).
- 회색 팔면체: 목표물 정답 위치(교수자 확인용, 체크박스로 끌 수 있음). 보고에 성공하면 초록색이 됩니다.
- 흰 선과 흰 점: 에이전트가 `debug()`로 보낸 경로·목표점입니다.

## 3. 센서

에이전트에게는 카메라 영상을 주지 않습니다. 뷰어의 1인칭 화면은 사람이 보는 용도입니다. 사양은 실제 제품 자료를 기준으로 잡았고, 숫자는 모두 `cavesim/config.py`에 있습니다. 센서는 기체에 고정돼 있어서, 드론이 가속하려고 기울면(최대 25°) LiDAR와 거리센서도 같이 기울어집니다.

| 센서 | 기준 | 주요 사양 | `Observation` 필드 |
|---|---|---|---|
| 3D LiDAR | Ouster OS0-32 (DARPA SubT 우승팀 CERBERUS 드론 탑재 기종) | 32채널, 수직 ±45°, 방위 512칸, 10 Hz, 최소 0.3 m. 거리 잡음 1~2 cm(15 m 넘으면 5 cm). 멀수록, 벽을 비스듬히 볼수록 반사가 빠짐 | `lidar`, `lidar_ranges`, `lidar_dirs` |
| IMU | ADIS16448 (RotorS/Gazebo 기본값) | 240 Hz. 백색잡음, 켜질 때 생기는 바이어스, 천천히 변하는 바이어스. 가속도계는 중력을 포함한 비력을 잼 | `imu` (틱마다 24개 샘플) |
| 하방 거리센서 | Benewake TFmini Plus | 0.1~12 m, 잡음 3 cm(5 m 넘으면 거리의 약 0.6%) | `range_down` |
| 기압계 | BMP388급 | 이륙 지점 기준 고도, 잡음 0.1 m + 시간에 따라 쌓이는 드리프트 | `baro_alt` |
| 자세 | 비행제어기 추정값 | 롤·피치, 잡음 0.3° 정도 | `roll`, `pitch` |
| 오도메트리 | 기체 내장 LiDAR-관성 오도메트리(LIO) 출력을 흉내 낸 값 | 이동 거리와 시간에 따라 위치와 방향 오차가 쌓임. `--odom off`면 위치와 속도를 주지 않음 | `pos`, `vel`, `yaw` |

LiDAR 점을 오도메트리 좌표로 옮길 때는 롤·피치까지 반영해야 합니다. `cavesim.api.body_to_odom(obs, obs.lidar)`를 쓰면 됩니다.

`--odom off`로 실행하면 IMU와 LiDAR만으로 위치를 직접 추정해야 합니다. 상위 과제나 가산점 과제로 쓸 수 있습니다.

`os0-32` LiDAR는 드론 1대당 스캔 한 번에 약 20 ms가 걸려서, 드론 3대면 실시간보다 약간 빠른 정도가 한계입니다. 느린 PC나 반복 실험에는 `--lidar lite`를 쓰세요.

## 4. 규칙

- **임무 시간** 600초, **배터리**는 제자리 비행 기준 480초입니다. 빨리 날수록 조금 더 빨리 닳습니다.
- **GPS 없음**: 에이전트는 실제 위치를 받지 못합니다. 받는 것은 아래 센서 값과 목표물 탐지기, 통신 메시지입니다.
- **충돌**: 벽이나 다른 드론에 1 m/s보다 빠르게 부딪히면 추락합니다. 살짝 스치는 정도는 괜찮습니다.
- **통신**: 거리 25 m 이내이면서 시야가 트여 있어야 합니다. 기지(원점)도 같은 조건입니다. 메시지는 다음 틱에 도착합니다.
- **보고**: `Command.reports`로 보고하며, 그 순간 기지와 직접 통신이 돼야 받아들여집니다. 위치는 오도메트리 좌표로 보내고, 정답에서 5 m 안이고 종류가 같으면 인정됩니다. 보고 횟수는 목표물 수의 2배로 제한됩니다.
- **귀환**: 임무가 끝났을 때 기지에서 수평 3 m 안에 착륙해 있으면 귀환으로 칩니다.
- **점수** = 목표물 10점 × 개수 + 귀환 5점 × 대수 + 탐사율 × 20

## 5. 동굴 지형과 공기

동굴 천장에는 종유석, 바닥에는 석순이 있고 넓은 방에는 둘이 이어진 석주도 있습니다. 모두 실제 충돌 형상이라 부딪히면 벽과 똑같이 추락하고 LiDAR에도 잡힙니다. 끝과 반대편 면 사이에는 항상 0.9 m 이상 공간을 남겨서 막힌 통로는 없지만, 그 사이를 지나려면 속도를 줄이고 높이를 맞춰야 합니다.

벽 가까이에서는 공기도 드론을 흔듭니다. 프로펠러 바람이 바위에 부딪혀 되돌아오기 때문입니다.

| 효과 | 내용 | `default` 기준 크기 |
|---|---|---|
| 동굴 기류 | 어디서나 부는 약한 돌풍 | 0.25 m/s² |
| 벽 난류 | 벽에서 1.2 m 안으로 들어가면 돌풍이 커짐 | 벽에 닿기 직전 최대 1.75 m/s² |
| 천장 효과 | 천장에서 0.8 m 안이면 천장 쪽으로 빨려 올라감 | 최대 중력의 35% |
| 지면 효과 | 바닥에서 0.6 m 안이면 양력이 늘어남 | 최대 중력의 15% |

비행제어기는 명령한 속도를 지키려고 하지만 교란이 생긴 뒤에야 바로잡습니다. 그래서 벽이나 종유석에 바짝 붙어 빠르게 날면 밀려서 부딪힐 수 있습니다. 장애물과 얼마나 떨어져서, 얼마나 빠르게 날지는 팀이 정해야 합니다. 에이전트는 교란 값을 직접 받지 못하고 IMU와 속도 추정으로만 알아챌 수 있습니다.

`--aero off`는 교란을 모두 끄고, `--aero hard`는 더 세게 합니다. 기본 에이전트(`agents.explorer`)는 `hard`에서 대부분 추락합니다. 값은 `cavesim/config.py`의 `AeroParams`에 있습니다.

## 6. 에이전트 작성

`agents/template.py`를 복사해서 시작합니다.

```python
from cavesim.api import Agent, Command

class MyAgent(Agent):
    def step(self, obs):
        # obs.pos, obs.yaw  : 추정 위치·방향 (오도메트리 좌표)
        # obs.roll, pitch   : 기체 기울기
        # obs.lidar         : (N,3) 장애물 점, 기체 좌표 (body_to_odom으로 변환)
        # obs.imu           : (24,7) IMU 원시 샘플
        # obs.detections    : 시야에 들어온 목표물
        # obs.inbox         : 받은 메시지
        return Command(vel=(1.0, 0, 0), yaw_rate=0.0)

Agent = MyAgent
```

```
python -m cavesim.run --agent agents.my_agent
```

에이전트가 받는 값(`Observation`)과 보낼 수 있는 명령(`Command`)은 모두 `cavesim/api.py`에 정의돼 있습니다. 시뮬레이터에서 학생이 읽어야 하는 파일은 이것 하나입니다.

여러 번 돌려 성능을 비교할 때는 뷰어 없이 Python에서 바로 호출합니다.

```python
from cavesim.sim import run_mission
from agents.explorer import Agent
print(run_mission("maps/medium_1", Agent, n_drones=3, odom="hard", seed=1))
```

## 7. 폴더 구조

```
cavesim/        시뮬레이터 (학생은 수정하지 않음)
  api.py        Observation / Command / Agent 정의
  sim.py        물리, 공기 교란, 통신, 채점
  sensors.py    센서 모델 (LiDAR, IMU, 거리센서, 기압계)
  cavegen.py    동굴 맵 생성기 (종유석·석순 포함)
  config.py     모든 상수 (속도 한계, 센서 사양 등)
  run.py        실행기 + 뷰어 서버
  check.py      설치 확인
agents/         에이전트 (학생 작업 공간)
  explorer.py   기본 에이전트: 반응형 탐사 + breadcrumb 귀환
  template.py   새 에이전트 시작점
viewer/         Three.js 뷰어
maps/           생성된 맵 (truth.json은 채점용 정답)
tests/          자동 테스트 (pytest)
```

## 8. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `pip install pybullet`이 컴파일 오류로 실패 | PyPI에는 Windows용 PyBullet이 없습니다. pip 대신 1장의 conda 방식으로 설치하세요. |
| `mamba env create`가 파일을 못 찾음 | 경로에 한글이 있습니다. 영문 경로(예: `C:\work`)로 옮기세요. |
| `conda env create`가 `conda-launchers` 관련 오류로 실패 | conda 26.x의 버그입니다. `mamba env create`를 쓰세요. 설치 스크립트는 이미 mamba를 씁니다. |
| Python이 아무 메시지 없이 종료됨 (Windows 오류 코드 `0xC06D007F`) | numpy가 MKL 2026.x를 쓰면 일부 PC에서 행렬 곱셈 중에 죽습니다. `environment.yml`이 OpenBLAS로 고정해 두었으니 `mamba env update -n cave-drone -f environment.yml -y`로 다시 맞추세요. |
| 뷰어가 "연결 끊김" | 실행기(`python -m cavesim.run`)가 꺼졌습니다. 터미널의 오류를 확인하세요. |
| 드론이 모델이 아니라 색 공으로 보임 | 3D 모델 파일이 없습니다. [2-4](#2-4-드론-3d-모델-선택)를 보세요. |
| 포트 8000 또는 8765가 사용 중 | `--http-port 8001 --ws-port 8766`처럼 다른 포트를 지정하세요. |
