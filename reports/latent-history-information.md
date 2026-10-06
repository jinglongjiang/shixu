# Latent and Historical Information

Hidden information and circle-generator shortcuts are separate findings. Recovery or prediction accuracy alone is not validated navigation improvement.

## Archive Policy

This is a classification-only consolidation. Original stage text, numbers, negative results and withdrawn claims are preserved byte-for-byte below. Later closure reports supersede earlier proposed next steps. The historical README is retained in thematic sections. Snapshot variants are labeled by their original paths. Frozen protocols and autoresearch_nav/program.md are not changed.

## Stage Index

- [LATENT_TEMPORAL_INFORMATION_AUDIT.md](#stage-1)
- [CIRCLE_HISTORY_CAUSAL_AUDIT.md](#stage-2)
- [README.md (historical README section 2)](#stage-3)
- [README.md (historical README section 3)](#stage-4)
- [outputs/forecast_control_remote_backup_20261004/README.md (historical README section 2)](#stage-5)
- [outputs/forecast_control_remote_backup_20261004/README.md (historical README section 3)](#stage-6)


---

<a id="stage-1"></a>

## Source: LATENT_TEMPORAL_INFORMATION_AUDIT.md

Original full-source SHA-256: dc1038b0320b89d3ed691651a77a528b0c3c9d9816fc5405fe49513441bff291

<!-- BEGIN PRESERVED SOURCE -->
# Latent Temporal Information Audit

日期：2026-10-04。范围：本地 shixu 的原生5人 circle/square 环境；离线信息恢复和小规模动作后果诊断。**没有训练导航新模型，没有启动KDA V9，没有修改环境、奖励或已有checkpoint。**

## 1. 当前结论

1. **历史有可测的信息增量，但不是跨场景稳定的长历史优势。** 在circle留出轨迹上，24帧探针优于当前帧探针；简单历史统计也能获得相近或更好的恢复结果。circle训练后的长历史探针在square上明显退化，CV很强。
2. **隐藏目标具有局部决策价值。** 在8个近身相遇状态中，补充真实目标相对当前帧估计能减少部分剩余到达时间；只补ORCA平滑状态没有稳定改善。这是离线oracle结果，不是可部署新方法的成绩。
3. **circle里的这部分收益可以非常简单。** 保存合法首次看见的位置，利用circle的对径目标规则推算目标，在4个近身circle状态、两种continuation下均达到真值参考的最高回报。只需两个位置数，不需要复杂关联记忆。
4. **没有得到KDA研发重新启动的证据。** 也没有证明整个时序方向没有价值。下一步应区分可泛化运动隐状态与生成器特有的起点—目标规则，而不是改门控或继续V9/V10。

## 2. 实际执行与冻结条件

| Split | 原生几何 | Cases | Episodes | 拟合/评估样本 |
| --- | --- | --- | ---: | ---: |
| Train | circle | 60000–60063 | 64 | 3,151 |
| Validation | circle | 60064–60079 | 16 | 693 |
| Test | circle | 60080–60111 | 32 | 1,362 |
| OOD test | square | 61000–61031 | 32 | 1,475 |

144条轨迹、7,138个控制帧；127次到达、14次碰撞、3次超时，全部保留。收集策略是继承的合法观测ORCA teacher，不是成功轨迹筛选。固定间隔采样，不按预测误差挑帧。每个case只属于一个split，禁止随机拆帧泄漏同一轨迹。

输入是合法可见/保留轨迹的位置、速度、半径、age和测量有效性，以及当前机器人状态和当前其他合法行人。数值ID不进入探针。隐藏目标、上一控制步的 `_last_pref_vel`、未来真实位置仅作为标签。历史截止当前帧。

当前帧基线不是只看单个人：它也接收当前其他合法行人的状态。每个窗口臂使用相同233维输入布局、256个RBF landmark、同一ridge模型族，3,912个回归系数加截距；未使用的过去槽位置零。统计臂为18维历史摘要加当前特征，2,584个系数加截距。landmark行号相同，只在训练集标准化；16个超参数组合只用validation选择。

统计臂包含合法首次看见位置、窗口首个测量、速度均值/标准差、速度与位置变化、测量数和速度范围。首次看见位置是生命周期信息，**不限定在24帧内**。窗口臂最多24帧，即覆盖首尾间5.75秒。

目标方向从合法当前位置指向真实目标；真实目标距离小于0.25m时不计算方向角误差。为获得统一的1/2秒未来标签，预测探针排除距轨迹终止不足2秒的帧，因此不能代表全部临碰撞尾部。

## 3. 隐藏状态恢复与运动预测

下表为逐样本平均；2秒运动误差是endpoint displacement error，不是导航效果。

| 输入/规则 | circle目标角误差° | circle平滑状态误差m/s | circle未来2s误差m | square目标角误差° | square平滑状态误差m/s | square未来2s误差m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 10.16 | 0.1783 | 0.3156 | 25.74 | 0.2029 | 0.3964 |
| History3 | 9.54 | 0.1754 | 0.2974 | 26.42 | 0.2122 | 0.3749 |
| History6 | 9.98 | 0.1770 | 0.2868 | 26.68 | 0.2309 | 0.3851 |
| History12 | 9.51 | 0.1728 | 0.2710 | 27.06 | 0.2514 | 0.4031 |
| History24 | 8.51 | 0.1548 | 0.2615 | 35.45 | 0.3387 | 0.4694 |
| Simple history statistics | 4.02 | 0.1276 | 0.2655 | 47.44 | 0.4464 | 0.5848 |
| Current velocity/CV | — | 0.1774 | 0.3246 | — | 0.0570 | 0.2493 |
| Circle first-sighting goal prior + CV | 1.82 | 0.1774 | 0.3246 | 52.53 | 0.0570 | 0.2493 |

CV行以当前实测/保留速度估计平滑状态、线性外推未来；**不声称该速度就是ORCA内部状态**。first-sighting规则的目标估计为首次合法观测位置的相反数，不读取真实出生位置；它是circle生成器专用规则，square一列是其边界检查。

circle目标位置误差：Current 1.0903m、History24 0.6343m、统计摘要0.2583m、首次看见位置规则0.0936m。这表明大量可恢复目标信息很可能来自生成器规律；不能把这一现象解释成复杂交互记忆。

按case先平均再配对bootstrap，Current减History24的2秒误差为 **+0.0475m，95%区间[0.0365, 0.0581]**；square为 **−0.0824m，区间[−0.0970, −0.0671]**。这与逐样本表使用不同权重。区间为探索性结果，没有多重比较校正。

circle保留遮挡样本221个：2秒误差Current 0.4909m、History24 0.3928m、统计0.3853m、CV 0.5346m。square保留遮挡样本146个：对应0.7373/0.7065/0.8350/0.5876m。不能把circle的增量推广到所有遮挡场景。

square的1秒误差Current 0.1437m、History24 0.1866m、CV 0.0806m。**更复杂的探针不是天然更强的baseline。**

## 4. 动作后果检查：不是持续一个动作数秒

每个root枚举原生80个动作，**只执行一个0.25s控制步**，随后执行同一个合法ORCA continuation到原生终止。另一套continuation仅关闭机器人teacher的TTC brake，行人动力学保持不变。根部teacher平滑状态由合法过去观测重建。

实际参考是该continuation下的折扣原奖励 `Q^pi`，gamma=0.99，progress reward仍为0。没有用最大进度当正确动作，没有声称得到最优Q。root动作不加新的风险评分、过滤或平滑；该诊断不是整套学习导航器的部署成绩。

Current/History24/Statistics分别估计行人的隐藏目标与上一时刻平滑状态，用同一个原生动力学模型产生预测世界。预测世界选动作，**真实世界只评估该动作**。CV单独使用恒速行人世界。truth只作为离线参考。

预测世界采用单点隐状态估计，没有传播后验；这一估计器失败不等于所有history-aware planning都失败。每个root所有人可见，以避免偷偷补入当前隐藏位置；因此它隔离的是潜在运动状态，不直接检验遮挡期行动。

### 4.1 预先固定的主诊断

Cases：circle 60080–60083、square 61000–61003；每case首次满足时间≥6s、五人全部可见、距机器人目标>1m的帧。均有合法root，无替换。

| 方法 | 原continuation平均regret | 关闭brake平均regret |
| --- | ---: | ---: |
| CV | 0.00206 | 0.00000 |
| Current probe | 0.01700 | 0.01702 |
| History24 probe | 0.00958 | 0.00939 |
| Statistics probe | 0.02216 | 0.02118 |

regret为真实 `max_a Q^pi - Q^pi(a_selected)`。8个root的80个动作在两套真实continuation下全部成功；CV在16个root×continuation组合中15个达到最高回报。因此这批root几乎没有安全改进空间，**不能据此否定整个benchmark的时序headroom**。

### 4.2 补充的近身相遇诊断

主诊断发现覆盖较容易的状态后，新增一个明确标记的探索性cohort，原结果不删除、不替换。仅按当前合法几何选状态：时间≥2s、五人全可见、距目标>1m、最近人体表面距离≤0.8m。每个case取首次合格帧，每种几何取case编号最小的4个合格case。**没有按方法结果、未来真值或终止类型筛选。**

Roots：circle 60080@14、60082@36、60084@30、60085@8；square 61000@13、61001@23、61002@16、61003@15。`@`之后为控制帧序号。60082@36与主诊断重复，不能当独立证据再加权。

下表为近身root的平均regret，越小越好；每列仅4个root。

| 方案 | circle原continuation | circle关闭brake | square原continuation | square关闭brake |
| --- | ---: | ---: | ---: | ---: |
| CV | 0.01274 | 0.00879 | 0.02046 | 0.02042 |
| Current | 0.03898 | 0.03155 | 0.01649 | 0.01627 |
| History24 | 0.01487 | 0.01901 | 0.01838 | 0.02002 |
| Statistics | 0.02795 | 0.02614 | 0.01266 | 0.01436 |
| Current + true goal only | 0.00000 | 0.00000 | 0.00183 | 0.00000 |
| Current + true preferred state only | 0.02964 | 0.03155 | 0.01838 | 0.01846 |
| Legal first-sighting goal prior | 0.00000 | 0.00000 | 0.01623 | 0.01981 |

所有方案选出的动作都最终成功，没有已验证的SR/CR优势。square 61000中，80个参考动作有11个碰撞，其余近身root的80个参考动作全部成功。

History24相对Current：原continuation 1胜/2负/5平，关闭brake 2胜/2负/4平。相对CV：2胜/2负/4平、3胜/3负/2平。**没有稳定超越简单规则。**

还原真实目标后，circle平均剩余到达时间由8.4375→7.1875s、8.3125→7.3125s；square由6.625→6.125s、6.6875→6.125s。还原平滑状态没有稳定对应收益。这隔离出目标估计的局部效率价值，但该oracle不可部署。

合法首次看见位置规则在circle的8个root×continuation组合中全部达到参考最高回报；相对CV的平均剩余时间分别减少0.4375s和0.3125s。**同回报不等于同动作、同安全距离或同轨迹。** 该规则在square不能恢复这种优势。

## 5. 事实、推断、未知

**已证实事实：** 隐藏目标和平滑状态确实进入行人动力学；circle合法历史改善这些指定探针的恢复/预测；目标oracle改变部分动作后果；两数的circle规则即可实现局部oracle回报；长历史的square泛化失败；平滑状态oracle未提供稳定收益。

**有证据支持的推断：** 目前较明确的信息主要是目标/路线恢复，而不是“改向后有选择地擦除旧证据”；circle长历史收益有较强生成器先验成分。共享后续控制有能力修正根部动作，但没有独立实验确认这是控制频率造成的。

**尚未证明：** 任意最强Current模型仍无法恢复目标；历史在一般square场景能稳定补上该缺口；隐藏目标恢复能提升原学习导航器SR；KDA比GRU或简单目标滤波更适合；ORCA内部状态在所有场景没有价值。当前数据不支持这些更强结论。

## 6. 下一步与停止条件

2026-10-04更新：后续受控审计已完成，见CIRCLE_HISTORY_CAUSAL_AUDIT.md。原始数字保留不变，但“square长历史泛化失败”必须限定为本轮circle拟合的探针；不能据此断言square缺少历史信息。新审计在相同新square测试样本上比较冻结拟合，circle-fit History24的2s误差为0.4876m，square-fit为0.2698m；原生circle首次位置目标捷径由源码与对照干预确认，非对径场景仍有历史预测增益，但没有稳定超越简单基线的导航收益。此前将OOD失败用于否定时序信息本身的过强解释撤回，不改变原实验事实。

本轮实测结束，不自动训练新导航架构。推荐下一步仅做**跨生成器的目标/运动隐状态恢复**：使拟合数据包含原生square，采用相对运动表征，并保留完全未用作选择的case；与CV、短窗统计和简单目标滤波比较，再接同一root-step continuation诊断。新的拟合/验证边界必须重新固定。

若历史仅学会对径目标规则，或所有动作收益均被合法简单统计吸收，不再用该结果支撑复杂记忆。若合法历史在非规则目标场景同时改善预测和原奖励后果，才有依据开展统一IL→RL的GRU/KDA竞争；无需先证明GRU必然失败。

## 7. 验证与复现

本轮全程本地4个CPU worker，没有下载数据、没有占用4090服务器磁盘。新增诊断没有修改冻结的科学源码；其SHA256仍为：

fb1cfdc5abda8b0af9df66aeae4bd4d1674894ea6b43eb5c55de3265f6889bd5

100项测试，97通过、3项原有可选legacy资产测试跳过。收集器去掉未使用的随机导航网络后，case60080的全部动作、奖励和合法观测与原始采集逐项完全一致。所有root单步恢复的最大状态差≤4.77e-7，奖励差为0。

主诊断6,400个分支；近身五臂6,400个分支；两个单隐变量oracle追加2,560个；合法目标规则追加1,280个。**合计16,640个候选分支，不是16,640个独立benchmark episodes**；动作证据来自10个case的15个不同root、两种continuation。原始逐动作return/terminal/distance/time均保留，不能按分支数夸大统计样本量。

数据和拟合参数在本地outputs/latent_information，不上传大文件。主要原始SHA256：

| Artifact | SHA256 |
| --- | --- |
| episodes.pt | c23bac187c537527a067ad0863ac7f5d9040ea5e90534941f040a31fadc2904e |
| probes.pt | e5718119f99c7c35a3f3970f37d1d2cd3a0d5b97163859e61f3baab8575164b8 |
| interaction_birth_decision_branches.pt | 045f84746aa2bd7b85d2feb4d83e8147679db80b8930997fb78a42cbf12b0a68 |

探针依赖是可选的audit extra，不进入导航runtime。本地执行环境为NumPy1.23.5、SciPy1.3.3、scikit-learn1.0.2、PyTorch2.1.0，使用/usr/bin/python；未在别的版本组合上声称复现。以下顺序从固定protocol复现；保留原始结果时应使用不同的output目录，避免覆盖。

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.latent_information collect --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_information probe
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --components --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --birth-prior --workers 4
```

<!-- END PRESERVED SOURCE -->


---

<a id="stage-2"></a>

## Source: CIRCLE_HISTORY_CAUSAL_AUDIT.md

Original full-source SHA-256: f198ac02a713a1076c20b6f4a6b846ab6e62ec13ca3f665d8abbc88898e259b8

<!-- BEGIN PRESERVED SOURCE -->
# Circle History: Generator Coupling and Residual Temporal Information

Date: 2026-10-04. Completed offline controlled audit; not KDA V9, navigation-policy training, or a new SR claim.

## 1. Direct Answer

**The proposed either/or is false. Circle has a provable generator shortcut, and observations beyond the current frame also contain useful predictive information. The latter has not produced a stable additional navigation benefit in this audit.**

Three distinct conclusions follow:

1. **Generator-specific goal memory is established.** For native-circle pedestrians visible at reset, two legal first-sighting coordinates determine the hidden goal exactly. Extra ordered history cannot add information about that already-determined goal. The legal first-sighting rule attains the true-world maximum return at all 16 selected circle roots under both continuations.
2. **Not all historical prediction gains are an antipodal-goal artifact.** After breaking that coupling while retaining starts and the goal set, historical observations still improve held-out goal and motion prediction. The same is true when fitting and testing within native square.
3. **A need for complex ordered memory is not established.** In circle, adding 24-frame history beyond the first-sighting anchor has practically equivalent goal error; short-window and summary controls have practically equivalent two-second prediction error. None of the three tasks shows a stable navigation advantage for ordered history over the simple controls. These are task/probe-specific conclusions, not a proof that all temporal methods are useless.

Remembering an initial observation is genuine memory, not fabricated evidence. What fails is the inference from this particular benefit to generalizable, complex temporal modeling or KDA necessity.

## 2. Evidence Stronger Than the Previous OOD Comparison

### 2.1 Exact source property

The frozen native generator in vendor/crowd_sim/envs/crowd_sim.py, function generate_circle_crossing_human, calls:

```python
human.set(px, py, -px, -py, 0, 0, 0)
```

Consequently, if a pedestrian is first legally observed at reset, its first observed position B is its spawn position and its goal is exactly G = -B. This result follows from the generating code, not from statistical significance. It does not apply exactly to pedestrians first observed after they have moved, and it does not determine all ORCA internal state or future interactions.

On the 64 fresh native-circle test episodes:

- 290/320 pedestrian tracks were legally visible at reset: 90.625% of tracks.
- Those tracks contribute 2,846/3,061 eligible actor-frame prediction samples: 92.976% of samples.
- All 2,846 corresponding goals are recovered within 1e-6m; maximum numerical discrepancy is 2.35e-7m.

The track and actor-frame denominators are different. The earlier progress update's roughly 93% refers to eligible actor-frame samples, not 93% of unique pedestrians.

### 2.2 Controlled goal-ownership intervention

Three tasks are evaluated separately:

| Regime | Initial positions | Goal set | Goal ownership | Dynamics / reward / observations |
| --- | --- | --- | --- | --- |
| Native circle | Native | Native | Own antipodal goal | Frozen parent |
| Circle-permuted | Identical paired native starts | Identical paired native set | Rotated, no actor keeps its own goal | Frozen parent |
| Native square | Native square | Native square | Native square | Frozen parent |

The rotation is fixed by case ID before motion. All 144 circle/circle-permuted pairs pass exact start/set identity checks. Only goal ownership is intervened upon; changed routes and later interactions are consequences. Later states are not claimed to remain geometrically identical across regimes.

The antipodal rule then fails decisively: for pedestrians visible at reset, mean goal error is 5.192m in circle-permuted and 3.870m in square, with zero exact recoveries in either task. This verifies that the intervention removes the original shortcut.

### 2.3 Fixed splits and controlled probes

Each regime uses 64 fitting cases 72000-72063, 16 validation cases 72064-72079 and 64 fresh test cases 72080-72143. Total: **432 trajectories, 21,579 control frames**. All unsuccessful trajectories are retained; no test case is used to choose a model or hyperparameter.

| Regime | Frames | Goal / collision / timeout across all 144 collection trajectories |
| --- | ---: | ---: |
| Circle | 7,484 | 127 / 12 / 5 |
| Circle-permuted | 6,959 | 137 / 1 / 6 |
| Square | 7,136 | 137 / 1 / 6 |

These are collection-teacher outcomes, not performance comparisons between the predictive probes.

Probe inputs contain only legal robot/current-crowd observations, legal actor history and, where enabled, the first legal sighting. Goals, previous ORCA preferred velocity and future positions are labels only. Track IDs are association keys, not numeric learned features. Cases and regime labels are not model inputs.

All seven probe arms use the same padded 235-dimensional layout, 256 shared fitting-row landmarks, eight regression outputs, 3,928 fitted coefficients plus eight intercepts, and the same validation grid. Each regime is fitted independently. Padding equalizes the coefficient layout, not effective feature complexity or every optimization difficulty.

| Arm | Additional information beyond current crowd state |
| --- | --- |
| Current | None |
| Birth | Two first-sighting position coordinates |
| History24 | Previous 23 actor rows, in arrival order |
| Birth + History3 | Birth and previous two actor rows |
| Birth + History24 | Birth and previous 23 actor rows |
| Birth + Bag24 | Birth and the same previous rows canonically sorted by content |
| Birth + Statistics | Birth and 16 cheap history-summary features |

Bag24 removes explicit arrival order, not every possible temporal cue: positions, velocities and observation ages can indirectly reveal order. Statistics also contain temporal summaries. Neither is a strict information-free control.

Future labels require another two seconds of archived trajectory, excluding terminal tails. Results concern these eligible samples, not every failure-critical frame.

## 3. Held-Out Prediction Findings

Tables show pooled actor-frame means. Confidence intervals below instead give equal weight to each test case and resample cases, not correlated frames. Each interval uses 5,000 paired bootstrap draws and 98.3333% confidence, correcting across three regimes for that contrast/endpoint; this is not a blanket correction over all reported analyses.

### 3.1 Goal recovery

| Regime | Current error (m) | Birth error | History24 error | Birth + History24 error |
| --- | ---: | ---: | ---: | ---: |
| Circle | 1.241 | 0.316 | 0.791 | 0.331 |
| Circle-permuted | 1.150 | 1.149 | 0.928 | 0.944 |
| Square | 1.333 | 1.311 | 1.106 | 1.116 |

| Case-weighted error reduction | Circle | Circle-permuted | Square |
| --- | --- | --- | --- |
| Current minus Birth | +0.859 [0.758, 0.968] | +0.010 [0.002, 0.019] | +0.040 [0.020, 0.061] |
| Current minus History24 | +0.490 [0.420, 0.561] | +0.171 [0.124, 0.227] | +0.187 [0.145, 0.230] |
| Birth minus Birth + History24 | -0.010 [-0.037, 0.016] | +0.156 [0.110, 0.208] | +0.144 [0.104, 0.184] |

The prospectively specified local goal-error margin is 0.1m. In circle, Birth and Birth + History24 meet practical equivalence on this endpoint; their interval lies entirely inside +/-0.1m. In both non-antipodal tasks, history beyond Birth improves goal recovery by more than 0.1m even at the lower confidence bound.

**Established interpretation:** simple anchor memory accounts for the tested circle goal-recovery benefit without requiring a long ordered sequence. Nevertheless, additional observed history remains predictively useful when this exact rule is removed. This does not prove that every stronger current-only estimator would fail.

### 3.2 Future motion and the ordered-history boundary

| Regime | Current 2s error (m) | History24 | Birth + History24 | Birth + Bag24 | Birth + Statistics | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Circle | 0.308 | 0.298 | 0.262 | 0.273 | 0.267 | 0.304 |
| Circle-permuted | 0.313 | 0.257 | 0.260 | 0.271 | 0.285 | 0.222 |
| Square | 0.298 | 0.270 | 0.273 | 0.279 | 0.271 | 0.252 |

Current minus History24 case-weighted 2s-error reductions:

- Circle: +0.0224m [0.0042, 0.0377].
- Circle-permuted: +0.0386m [0.0266, 0.0571].
- Square: +0.0205m [0.0118, 0.0295].

The positive improvements in the non-antipodal tasks refute attributing **all observed-history prediction value** to G = -B. However, history does not beat the strong CV forecast there on the pooled endpoint.

The local 2s-error equivalence margin was fixed at 0.02m. In circle:

- Birth + Bag24 minus Birth + History24: +0.0118m [0.0051, 0.0187], statistically positive but inside the practical-equivalence margin.
- Birth + Statistics minus Birth + History24: +0.0047m [-0.0059, 0.0139], practically equivalent.
- Birth + History3 minus Birth + History24: +0.0016m [-0.0142, 0.0142], practically equivalent.

Thus a measurable order effect can coexist with practical equivalence. Nonsignificance is not used as evidence of equivalence. In square, all three simple controls also meet this endpoint's equivalence criterion against Birth + History24. Circle-permuted equivalence is established for Bag24 but unresolved for Statistics and History3. No universal ordering-independence claim follows.

### 3.3 Correction of the earlier square interpretation

An additional, explicitly post-hoc evaluation applies already-frozen fits to the **same** fresh square test samples:

| Fitting regime | Current 2s error | History24 2s error |
| --- | ---: | ---: |
| Circle | 0.3819m | 0.4876m |
| Square | 0.2977m | 0.2698m |

The circle-fit minus square-fit History24 difference is +0.2025m [0.1767, 0.2289] with equal case weighting. No test-driven refitting was performed.

**Correction:** the prior circle-trained/square-tested failure establishes a transfer failure of that predictor, not an intrinsic absence of useful history in square. Using it to conclude that historical information itself cannot generalize was too strong and is withdrawn. These results concern the offline probes, not a demonstrated cause of the old KDA IL/RL failures.

## 4. Actual Action Consequences, Not Only Prediction Error

Selection was fixed before method outcomes: the first eligible root in each case, first 16 eligible test cases per regime, time >=2s, all five pedestrians visible, robot goal distance >1m and minimum surface clearance <=0.8m. Total: **48 distinct roots**. All root one-step snapshot restores match archived states within 4.77e-7 and exactly match rewards.

For each root, nine prediction/reference worlds enumerate the same native 80 commands. The root command executes for **one 0.25s step**; all candidates then share a legal ORCA continuation to native termination. A second continuation only disables the robot teacher's TTC brake. Human dynamics, action support and original reward remain frozen; gamma=0.99 and progress_reward=0.

A predicted world selects the action; its return is evaluated in the corresponding true world. Regret is max_a Q^pi(s,a) - Q^pi(s,a_selected), not Q*. Estimated worlds replace only the hidden goal and previous preferred velocity; CV uses constant-velocity humans. The first-sighting rule combines its goal rule with the legally current velocity, not true preferred state.

**69,120 candidate branches are computation, not 69,120 independent episodes.** Statistical units are the 16 roots per regime. Both continuations reuse each root and are sensitivity checks, not independent replications.

### 4.1 Mean native-return regret

| Regime / continuation | Current | Birth | History24 | Birth + History24 | Birth + Bag24 | Birth + Statistics | CV | Circle birth rule | Truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Circle / inherited | .00189 | .00202 | .00545 | .00186 | .00196 | .00253 | .00242 | .00000 | .00000 |
| Circle / unbraked | .00234 | .00143 | .00539 | .00238 | .00278 | .00243 | .00323 | .00000 | .00000 |
| Permuted / inherited | .03545 | .03637 | .03392 | .03392 | .03541 | .03638 | .02554 | .03039 | .00000 |
| Permuted / unbraked | .03034 | .03097 | .02988 | .03037 | .03028 | .03126 | .02103 | .02451 | .00000 |
| Square / inherited | .01161 | .00888 | .01111 | .01198 | .00099 | .00885 | .01170 | .02635 | .00000 |
| Square / unbraked | .01104 | .01900 | .01351 | .01431 | .00198 | .00931 | .00229 | .02526 | .00000 |

In native circle the legal birth rule achieves zero regret at **32/32 root-continuation combinations**. Identical return does not imply identical action or clearance. Some circle roots have unsafe alternatives: successful true-world commands range from 53/80 to 80/80. The result is not based solely on states where every command works.

Prespecified Birth minus Birth + History24 regret differences:

| Regime | Inherited continuation | Unbraked continuation |
| --- | --- | --- |
| Circle | +.00016 [-.00350, .00375] | -.00095 [-.00446, .00293] |
| Circle-permuted | +.00245 [.00000, .00542] | +.00060 [-.00574, .00522] |
| Square | -.00310 [-.01606, .00457] | +.00469 [-.00344, .02191] |

The local regret margin is 0.01. Circle and circle-permuted meet practical equivalence for these two probe arms under each continuation; square does not have enough precision to establish equivalence. Circle's remaining-time differences also meet the prospectively fixed +/-0.25s equivalence criterion. This is stronger than merely reporting nonsignificance, but remains local to the specified roots, probe family and continuations.

### 4.2 Safety, completion and simple alternatives

- Circle and square: every evaluated selector succeeds at 16/16 roots under each continuation, with zero collisions and timeouts. This small success-only cohort cannot prove SR/CR equivalence across the benchmark.
- Circle-permuted: all eight non-oracle selectors succeed at 15/16 roots and timeout on the same case, 72086; truth succeeds at 16/16. That root has only two successful true-world commands under inherited continuation and one under unbraked continuation. Latent full-information headroom exists there, but the tested history probes do not recover it.
- Native-circle rule versus truth mean remaining time: 7.4844s versus 7.4844s inherited, 7.6562s versus 7.6562s unbraked. Clearance is not identical; no safety-dominance claim is made.
- Birth + History24 versus CV regret wins/ties/losses: circle 3/10/3 and 2/13/1; circle-permuted 2/10/4 and 2/8/6; square 2/11/3 and 0/10/6. No stable history advantage over CV is demonstrated.
- A post-hoc case-paired comparison in square/unbraked finds Birth + History24 worse than CV by +.01202 regret [.00100, .03168]. This unfavorable result is retained, not selected away.
- In square/unbraked the prespecified Bag24 minus ordered-history difference is -.01233 [-.03165, -.00066]; the simpler content-sorted representation performs better in this diagnostic. That is evidence against an automatic benefit from arrival order, not proof that order is inherently harmful.

The action test applies a predictive latent-state consumer, not the learned Mamba-VL or KDA value head. It does not measure full-policy IL/RL performance. Roots are currently fully visible, so this test isolates hidden motion state rather than directly evaluating actions during occlusion.

## 5. What Is Settled, and What Is Not

| Question | Evidence-based answer |
| --- | --- |
| Does native circle contain an exact first-position/goal shortcut? | **Yes: source proof plus numerical verification.** |
| Does long ordered history add goal information for actors visible at reset, beyond their first-position anchor? | **No for that goal variable: it is already determined.** This is not a statement about every future-motion latent. |
| Is all usable history outside the current frame just that shortcut? | **No for the tested predictive estimators:** residual gains survive the coupling intervention and within-square fitting. Information-theoretic insufficiency of all possible current-only models is not established. |
| Does ordered long history improve the tested circle decisions beyond simple anchor memory? | **No additional practical benefit established; the tested probe contrast meets regret/time equivalence and the analytic rule is locally return-optimal.** |
| Does history establish an independent navigation advantage in the other tasks? | **No:** prediction improves, but simple controls absorb or outperform the observed decision gains. |
| Does this explain the original Mamba-VL no-history versus T=24 SR difference, or KDA V1-V8 losses? | **Not established.** Those are different trained-model comparisons; these probes do not causally decompose their SR. |
| Does this justify restarting KDA V9? | **No evidence-based justification yet.** Nor does it permanently reject KDA or all temporal modeling. |

**Operational conclusion:** native-circle goal recovery must not be used as the main justification for a complex temporal navigation mechanism. The appropriate claim is narrower: legal historical observations can improve latent/motion prediction beyond this generator rule, but a deployable navigation benefit beyond strong simple memory/CV remains unproved. No percentage of the old navigation SR gain can honestly be assigned to the shortcut from this audit alone.

## 6. Reproducibility and Frozen Boundaries

Only offline experiment helpers and tests are added. Existing production navigation sources, simulator, reward, checkpoints and V1-V8 results are unchanged. The optional audit dependency is not a deployment dependency. Local execution used four CPU workers, NumPy1.23.5, SciPy1.3.3, scikit-learn1.0.2 and PyTorch2.1.0; no GPU training or server downloads.

Frozen scientific source SHA256:

fb1cfdc5abda8b0af9df66aeae4bd4d1674894ea6b43eb5c55de3265f6889bd5

Protocol was saved with the collected data before fitting or decision outcomes. Its SHA256:

1a4034d2fa886e403293053d7e71e4cf2b51d2eb7820b19b3e13c080e4b4a8a7

Raw local artifacts remain in the ignored directory:

/home/abc/workspace/shixu/outputs/circle_coupling/

| Artifact | SHA256 |
| --- | --- |
| episodes.pt | fac9e6ad3be51c33ec2c62435e67f2ce49f368d619872fe111dfacd8c7dea1ea |
| models.pt | 2de129de8b94075ce1ad06a9805f0bfa7f8a5ea5e00e3626a0875f93819cb959 |
| decision_branches.pt | 89e623b39b51950e46a4156da1aae78e4e58168c74c37348401dbb6a69855b92 |
| probe_results.json | a2629ec95a7c17b1c062124d5525805a296fcc38357d18c1671e9a5aae338a14 |
| decision_results.json | ffc0d5a5032533bc3b3b40ece4ade08440acd03ba19e0c4e4149e7c4fa42d626 |
| cross_distribution_results.json | 9dcacd898e108dcb3aeb0154ac9630671bc673eb7b88cf02f75ce21592ca8cbe |

To reproduce without overwriting these archived results, use a different --output directory consistently:

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling collect --workers 4 --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling probe --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling cross --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling decisions --workers 4 --output outputs/circle_coupling_repeat
PYTHONPATH=vendor:. python -m unittest discover -s tests -v
```

Verification: 109 tests, 106 passed and three existing optional legacy-asset skips. Nine new tests check the intervention, legal feature separation, order control, shared input layout, case-paired inference and disjoint splits. No automatic new navigation training follows this audit.

<!-- END PRESERVED SOURCE -->


---

<a id="stage-3"></a>

## Source: README.md (historical README section 2)

Original full-source SHA-256: 0f5332ec89c79e6ab9fc605bb8e502e3aca61323ac48851518db15cb3f732d9a

<!-- BEGIN PRESERVED SOURCE -->
### Latent Information Audit After V8

The separate LATENT_TEMPORAL_INFORMATION_AUDIT.md records actual offline
hidden-goal/preferred-state probes and one-step-root native-reward continuations.
It is not KDA V9 or a new navigation-policy training run. All144 natural
five-person ORCA trajectories, including unsuccessful ones, are case-disjoint
across64 fitting,16 validation,32 circle test and32 square OOD episodes.

In circle,24-frame history reduces two-second motion error from0.3156m to0.2615m;
simple history statistics achieve0.2655m. In square, the same long-history probe
degrades from0.3964m current-only error to0.4694m, while CV achieves0.2493m.
Single-hidden-variable oracle controls identify local goal-estimation value,
not a stable benefit from restoring the ORCA preferred-velocity state. A legal
first-sighting position rule attains the reference maximum return at all four
close-interaction circle roots under both continuations, exposing a strong
generator-specific alternative to complex memory. This is not evidence for
a deployed-policy SR advantage, information-theoretic insufficiency of current
state, or KDA-family rejection. No new navigation training is started.

The predictive fits need the optional audit dependency (`python -m pip install
-e '.[audit]'`); it is not part of the navigation runtime. This local run used
NumPy1.23.5, SciPy1.3.3, scikit-learn1.0.2 and PyTorch2.1.0. Use the frozen
vendored simulator and an environment with these dependencies:

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.latent_information collect --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_information probe
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --components --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --birth-prior --workers 4
```

Artifacts remain in ignored outputs/latent_information. Candidate branches
share frozen roots and are not independent benchmark episodes. The report
retains the easy primary cohort, explicitly labels the added interaction and
component diagnostics as exploratory, and reports both continuations rather
than selecting the favorable one. Tests now total100, with97 passes and three
optional legacy-asset skips; the frozen scientific source hash is unchanged.


<!-- END PRESERVED SOURCE -->


---

<a id="stage-4"></a>

## Source: README.md (historical README section 3)

Original full-source SHA-256: 0f5332ec89c79e6ab9fc605bb8e502e3aca61323ac48851518db15cb3f732d9a

<!-- BEGIN PRESERVED SOURCE -->
### Controlled Circle-Coupling Audit

CIRCLE_HISTORY_CAUSAL_AUDIT.md separates first-sighting goal information from
residual history and explicit sequence order. It uses432 fresh trajectories:
native circle, paired circle with goal ownership permuted (identical starts and
goal set), and native square. Each regime has separate64-case fits,16-case
validation and64-case held-out evaluation. No navigation model is trained.

The native generator exactly sets each goal to minus its spawn position.
The legal first-sighting rule recovers goals exactly for the290/320 circle test
tracks visible at reset, and attains true-world maximum return at all16 selected
circle roots under both native continuations. History still improves prediction
after breaking this coupling and when fitted within square, but not navigation
over the strong simple controls. The earlier square OOD failure is therefore a
transfer failure of that fit, not proof that square has no useful history.

The action audit enumerates69,120 branches across48 distinct frozen roots;
branches are not independent episodes. It executes one0.25s root action followed
by the same legal continuation and measures native-return regret, completion,
clearance and remaining time. This is Q^pi, not optimal Q or learned-policy SR.
The report preserves unfavorable CV/order comparisons and states which local
equivalence criteria hold. Production sources and old checkpoints are frozen.

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling collect --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling probe
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling cross
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling decisions --workers 4
```

Raw artifacts remain local in ignored outputs/circle_coupling. Tests total109,
with106 passes and three existing optional legacy-asset skips. No KDA V9 or
new IL/RL run is started on the basis of this audit.


<!-- END PRESERVED SOURCE -->


---

<a id="stage-5"></a>

## Source: outputs/forecast_control_remote_backup_20261004/README.md (historical README section 2)

Original full-source SHA-256: d5ac9d5a389c9f6a37c0f9ddc216d8995770f37cc60863cfa476233835b54058

<!-- BEGIN PRESERVED SOURCE -->
### Latent Information Audit After V8

The separate LATENT_TEMPORAL_INFORMATION_AUDIT.md records actual offline
hidden-goal/preferred-state probes and one-step-root native-reward continuations.
It is not KDA V9 or a new navigation-policy training run. All144 natural
five-person ORCA trajectories, including unsuccessful ones, are case-disjoint
across64 fitting,16 validation,32 circle test and32 square OOD episodes.

In circle,24-frame history reduces two-second motion error from0.3156m to0.2615m;
simple history statistics achieve0.2655m. In square, the same long-history probe
degrades from0.3964m current-only error to0.4694m, while CV achieves0.2493m.
Single-hidden-variable oracle controls identify local goal-estimation value,
not a stable benefit from restoring the ORCA preferred-velocity state. A legal
first-sighting position rule attains the reference maximum return at all four
close-interaction circle roots under both continuations, exposing a strong
generator-specific alternative to complex memory. This is not evidence for
a deployed-policy SR advantage, information-theoretic insufficiency of current
state, or KDA-family rejection. No new navigation training is started.

The predictive fits need the optional audit dependency (`python -m pip install
-e '.[audit]'`); it is not part of the navigation runtime. This local run used
NumPy1.23.5, SciPy1.3.3, scikit-learn1.0.2 and PyTorch2.1.0. Use the frozen
vendored simulator and an environment with these dependencies:

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.latent_information collect --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_information probe
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --components --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 python -m experiments.latent_decisions --cohort interaction --birth-prior --workers 4
```

Artifacts remain in ignored outputs/latent_information. Candidate branches
share frozen roots and are not independent benchmark episodes. The report
retains the easy primary cohort, explicitly labels the added interaction and
component diagnostics as exploratory, and reports both continuations rather
than selecting the favorable one. Tests now total100, with97 passes and three
optional legacy-asset skips; the frozen scientific source hash is unchanged.


<!-- END PRESERVED SOURCE -->


---

<a id="stage-6"></a>

## Source: outputs/forecast_control_remote_backup_20261004/README.md (historical README section 3)

Original full-source SHA-256: d5ac9d5a389c9f6a37c0f9ddc216d8995770f37cc60863cfa476233835b54058

<!-- BEGIN PRESERVED SOURCE -->
### Controlled Circle-Coupling Audit

CIRCLE_HISTORY_CAUSAL_AUDIT.md separates first-sighting goal information from
residual history and explicit sequence order. It uses432 fresh trajectories:
native circle, paired circle with goal ownership permuted (identical starts and
goal set), and native square. Each regime has separate64-case fits,16-case
validation and64-case held-out evaluation. No navigation model is trained.

The native generator exactly sets each goal to minus its spawn position.
The legal first-sighting rule recovers goals exactly for the290/320 circle test
tracks visible at reset, and attains true-world maximum return at all16 selected
circle roots under both native continuations. History still improves prediction
after breaking this coupling and when fitted within square, but not navigation
over the strong simple controls. The earlier square OOD failure is therefore a
transfer failure of that fit, not proof that square has no useful history.

The action audit enumerates69,120 branches across48 distinct frozen roots;
branches are not independent episodes. It executes one0.25s root action followed
by the same legal continuation and measures native-return regret, completion,
clearance and remaining time. This is Q^pi, not optimal Q or learned-policy SR.
The report preserves unfavorable CV/order comparisons and states which local
equivalence criteria hold. Production sources and old checkpoints are frozen.

```bash
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling collect --workers 4
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling probe
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling cross
PYTHONPATH=vendor:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.circle_coupling decisions --workers 4
```

Raw artifacts remain local in ignored outputs/circle_coupling. Tests total109,
with106 passes and three existing optional legacy-asset skips. No KDA V9 or
new IL/RL run is started on the basis of this audit.


<!-- END PRESERVED SOURCE -->
