# Action Commitment Experiments

Finite multistep headroom and release effects were observed. These are distinct from final method validation: the subsequent bounded gate search ended in FINAL_CONFIRMATION_FAILED. No final-set retuning is authorized.

## Archive Policy

This is a classification-only consolidation. Original stage text, numbers, negative results and withdrawn claims are preserved byte-for-byte below. Later closure reports supersede earlier proposed next steps. The historical README is retained in thematic sections. Snapshot variants are labeled by their original paths. Frozen protocols and autoresearch_nav/program.md are not changed.

## Stage Index

- [multistep-headroom-decision-test.md](#stage-1)
- [online-action-commitment-v0.md](#stage-2)
- [online-action-commitment-v01.md](#stage-3)
- [commit-advantage-learnability.md](#stage-4)
- [AUTONAV_OVERNIGHT_REPORT.md](#stage-5)


---

<a id="stage-1"></a>

## Source: multistep-headroom-decision-test.md

Original full-source SHA-256: 2811373c97cfb39871edc2543b9288b3271099145b9f66f70358aea69cdc004d

<!-- BEGIN PRESERVED SOURCE -->
# 有限多步headroom决胜实验

更新：2026-10-05。结论：MULTISTEP_HEADROOM_CONFIRMED。

## 1. 直接结论

**4/4个固定fresh root存在短时连续控制挽救空间。** 这4个root此前各80种首动作均超时、原Q完全同分；本轮持续原生grid动作1秒或2秒后恢复原策略，全部找到了成功且最小间距不低于0.02米的序列。

这确认了“单步干预无终局收益，但短时多步干预有收益”的可重复现象。它不是新算法成功、总体SR改善、critic病因证明，也不意味着必须使用复杂规划或时序模型。

**一个必须正视的简单替代：持续原策略自己选出的grid动作2秒，已经挽救3/4个root。** 因此停止大范围问题发现，下一阶段进入有限的动作持续性方法原型；复杂序列搜索必须先证明相对这个简单替代还有增量。

## 2. 冻结合同

- 原root、checkpoint、critic、reward、观测、历史、filter、动力学、动作支持均不变。不训练。
- 80动作原生排列是heading-major：action = heading * 5 + speed，共16方向、5速度，不包含额外stop。
- 每个grid动作分别持续4步或8步，共4 root × 80 × 2 = 640条实际续跑。
- 每步执行 a_t = 0.3 a_(t-1) + 0.7 u，而不是重复一个已经平滑的命令，也不是二次平滑。
- 干预期间仍每0.25秒更新合法观测、tracker和真实历史。假想候选不写入记忆。遇到原生terminal立即停止。
- 干预结束后，用同一checkpoint的原标量predict续跑到原终止；不引入批量近似策略或新consumer。
- 原gamma=0.99，从root累计原任务reward。不把进度换成新奖励。到达时间同时保存剩余时间和episode绝对时间。
- 全部分支保留碰撞、超时、到达和实际最小间距。0.02米只是本轮成功筛选，不降低母体0.2米filter或修改碰撞标准。
- 强制干预绕过选择器，属于离线headroom诊断；逐步记录原CV filter是否会阻挡命令。不能把强制执行当成新安全策略。

协议在任何多步结果出现前冻结。备用root也提前从既有parent-only图谱选择，没有重新跑场景或依据新方法收益选样本。

## 3. 第一层结果

下表S/C/T为到达/碰撞/超时，每格共80条。所有到达分支均通过本轮0.02米筛选，没有被隐藏的fragile到达。

| 配置 / case / seed / root时间 | 单步S/C/T | 持续1秒S/C/T | 持续2秒S/C/T | 多步合格到达数/160 |
|---|---|---|---|---:|
| 5-square / 81000 / 491 / 7.25s | 0/0/80 | 13/2/65 | 29/4/47 | 42 |
| 10-square / 81001 / 419 / 8.25s | 0/0/80 | 11/0/69 | 27/3/50 | 38 |
| 5-square / 81003 / 419 / 2.00s | 0/0/80 | 0/0/80 | 1/2/77 | 1 |
| 10-square / 81006 / 443 / 11.25s | 0/0/80 | 13/0/67 | 22/4/54 | 35 |
| 合计 | 0/0/320 | 37/2/281 | 79/13/228 | 116 |

640条中：116到达、15碰撞、509超时。**116/640不是方法成功率**：这是枚举的候选控制空间，不是可部署选择器。碰撞分支证明不能不加判断地延长动作。

4个独立case覆盖5/10人、419/443/491三个训练权重，不是四训练种子统计，也不包含20人或circle的新增证据。81003仅一个2秒序列成功，其局部可行空间明显窄于其他三个root。

### 最佳合格序列

最佳按原折扣Q选取，所有动作编号从0开始。剩余时间从root起算，绝对到达时间从episode起算。

| case | grid动作及持续时间 | 原Q | 序列Q | Q增量 | 剩余到达时间 | 绝对到达时间 | 最小间距 |
|---|---|---:|---:|---:|---:|---:|---:|
| 81000 | 24 × 2s | -0.089658 | 0.817907 | 0.907565 | 5.25s | 12.50s | 0.203717m |
| 81001 | 34 × 2s | -0.093336 | 0.747172 | 0.840508 | 7.50s | 15.75s | 0.469526m |
| 81003 | 24 × 2s | -0.072598 | 0.623525 | 0.696124 | 12.00s | 14.00s | 0.380831m |
| 81006 | 29 × 2s | -0.105299 | 0.868746 | 0.974045 | 3.75s | 15.00s | 0.610302m |

最佳4条序列在全部32个干预步中，原filter均不会阻挡命令；实际续跑最小间距也全部超过0.2米。至少这些见证不依赖放松filter或毫米级擦边。

## 4. 简单替代与反例

这是已有640条结果的事后分组，不是新增实验或重新选择主判据。

| case | 原策略root grid动作 | 持续该动作1秒 | 持续该动作2秒 | 2秒分支Q | 2秒分支最小间距 |
|---|---:|---|---|---:|---:|
| 81000 | 34 | 到达 | 到达 | 0.801631 | 0.304997m |
| 81001 | 22 | 超时 | 到达 | 0.598956 | 0.227478m |
| 81003 | 60 | 超时 | 超时 | -0.072598 | 0.804804m |
| 81006 | 28 | 到达 | 到达 | 0.809728 | 0.694640m |

原root选择不变，仅延后重新选择，就挽救了3个case；这三个2秒分支的命令也未被原filter阻挡。**不能因此把全部问题写成“首动作选错”或“必须增加复杂规划”。** 尚未检查为什么原策略后续会偏离这些可行控制。

另一反例：动作24持续2秒在81000、81003成功，却在81001、81006碰撞，最小间距分别为-0.262659米和-0.028377米。四个root不存在共同的安全成功2秒grid动作。不能把固定方向推进包装成通用修复。

## 5. 裁决与停止分支

冻结判据是至少2个独立case有合格多步挽救、单步没有合格挽救。本轮达到4个，第一层已经通过。

- 第二层局部两阶段序列：未启动。无需增加896序列/root。
- 备用多步及单步实验：未启动。仅提前完成冻结和基线合同校验。
- 备用case为5-square/81004/443、10-square/81007/419、5-square/81010/419、10-square/81011/491；不是被测试后的替换样本。
- 不扩大80²、不新增场景、不训练、不重启KDA。
- 上轮单步错排独立确认未通过的结论仍保留。本轮是在新授权、不同干预时域下得到的新证据，不改写旧负结果。

## 6. 已知与未知

| 证据等级 | 内容 | 边界 |
|---|---|---|
| 已证实 | 固定4个root全部存在1–2秒干预后到达的控制序列 | 只涉及这些root及同一策略续跑 |
| 已证实 | 原选动作持续2秒已挽救3/4个root | 简单动作持续性是必须比较的强对照，不是总体安全收益 |
| 已证实 | 最佳4条序列无需放松原filter，间距均>0.2m | 不证明所有持有动作安全，也不证明filter始终正确 |
| 有证据支持的推断 | 单独优化root首动作不能利用本轮所见全部控制空间，持续控制值得做最小原型 | 不指认critic horizon、训练覆盖、时序记忆或后处理为唯一病因 |
| 尚未证明 | 如何仅靠合法当前/过去输入选择正确的动作、持续时间和启动时刻 | 最佳序列来自真实续跑枚举，在线方法不能使用这个oracle |
| 尚未证明 | 能否保护原本成功的episode，是否具有总体SR/安全/效率增量 | 本轮未测成功保护或部署选择器，不将失败挽救率外推 |
| 尚未证明 | 简单持续性是否留下足以开发新方法的稳定残余 | 不能从4/4 oracle挽救直接宣布方法立项成功 |

**root时刻尤其重要：** 旧规则是回看父轨迹，选第一段未来8秒低进度区间的起点。部署时不能提前知道之后8秒会低进度。因此本轮有合法观测和合法原生动作，但干预时刻由事后诊断指定；在线触发能力尚未得到证明。直接把这4个时刻做成线上触发，就是信息泄漏。

## 7. 唯一下一步

**进入一个有限的、合法在线的动作持续性原型，而不是新增记忆或critic。** 首先比较原策略与简单“原选动作持续1/2秒、逐步沿用原安全检查”对照；触发必须只读当前和过去信息，不能使用事后root、终局标签或oracle最佳动作。

下一协议必须先冻结在线触发、持续/中断规则和成功保护集，再进行共同闭环比较。只有简单持有留下可重复的安全/任务收益缺口，才研究额外的序列评价机制及最近先例。本轮不自动实现这一新策略。

## 8. 正确性与归档

- 8个主/备用root的基线回放全部通过，整段命令偏差0，最大原Q浮点误差2.78e-17。
- 校验全部640分支哈希、协议、planned sequence、实际命令、时长和结果对应关系。干预平滑递推最大误差0。
- 保存93,140个实际控制步的命令；每条分支保存干预期间原CV clearance和filter判断。
- 4个最佳序列逐条做确定性复放，命令、终局、原Q、时长、最小间距完全一致。这是正确性复放，不冒充独立case确认。
- 全仓库测试：200 passed、3 skipped、68条既有NumPy/SciPy警告。3项旧parity测试未配置原资产路径，不声称通过。
- 核心shixu/vendor和所有checkpoint未修改。当前提交仅新增诊断、测试、报告与证据。
- 主640条初次运行约146.7秒，包含并行进程调度；不是单次在线规划延迟，也不包含本轮编写/测试/归档耗时。
- 汇总初次因NumPy int64 JSON序列化失败，已单列implementation-correction.json。只将fragile计数转为Python int，协议/控制代码/640条结果不变；从缓存续接，无重跑、无覆盖。

报告：
/home/abc/workspace/shixu/multistep-headroom-decision-test.md

完整本地证据：
/home/abc/workspace/shixu/outputs/multistep-headroom-decision-test/

protocol.json、implementation-correction.json、primary-constant.json、summary.json和validation.json纳入Git；validation包含4条完整成功见证，primary-constant包含全部640条结果及原始分支哈希。完整640条续跑命令及既有checkpoint保留本地，不能宣称单独clone就是自包含复现。

## 9. 后续在线原型结果（2026-10-05）

ONLINE_ACTION_COMMITMENT_V0已在全新86000–86031完成1536个配对闭环episode。固定过去2秒净进展≤0.2米触发、原选grid动作持有2秒、逐步重查原filter的组合，square挽救14个失败、同时破坏15个原成功；整体SR82.81%→80.73%，CR2.60%→5.34%。冻结裁决为B_RESCUE_WITH_UNACCEPTABLE_DAMAGE，不继续调参救V0。

这不改写本报告4/4多步headroom和原动作持有挽救3/4的事实；它否定的是本次固定在线触发/持有/释放组合的合格净收益。持有期间27次碰撞出现在原filter的all-unsafe fallback下，是否提前释放即可改善仍未验证，不从该共现推断唯一根因。

完整在线报告：
/home/abc/workspace/shixu/online-action-commitment-v0.md

<!-- END PRESERVED SOURCE -->


---

<a id="stage-2"></a>

## Source: online-action-commitment-v0.md

Original full-source SHA-256: 05093a91782c2a2a074145a8afa0c28f7c850934fd526d02b68d518d5d9cd03d

<!-- BEGIN PRESERVED SOURCE -->
# 在线动作持续性V0：完整配对闭环结果

更新：2026-10-05。冻结裁决：B_RESCUE_WITH_UNACCEPTABLE_DAMAGE。

## 1. 直接结论

**固定“过去2秒进展不足→持续原选动作2秒”可以救回部分失败，但不是合格改进。** square救回14个失败episode，同时破坏15个原成功episode；circle救回4个、破坏19个。整体成功率下降2.08pp，碰撞率上升2.73pp。

旧多步headroom仍成立。失败的是这个可部署的固定触发/持有/释放组合，不能据此否定所有动作持续性或规划。当前也不能把收益与损失分别归因给触发器、2秒时长、fallback或critic中的唯一一个。

本轮不调阈值、不延长/缩短持有、不训练、不换模型。保留完整负结果，不自动开启V1。

## 2. 实际实现

新增一个独立控制状态机，继承原ValuePolicy；原评分器、critic、奖励、状态、80动作、观测/tracker/history、原filter与动力学文件未改。模型仍使用已训练的四个CV权重：419、443、467、491，各128条IL示范、IL50、在线MC3000。没有新训练。

唯一行为变化：

1. 保存9个真实控制帧的目标距离，覆盖过去2秒。
2. 当前距目标>1米，过去2秒净进展≤0.2米，并且原策略选择未被原hard filter屏蔽时，启动持有。
3. 固定当前原生grid index，最多执行8步；每步重新计算0.3旧实际命令+0.7 grid的平滑命令。
4. 每0.25秒更新合法观测、tracker和真实history。持有步不重复调用critic选择，但每步重算原hard filter。
5. 原hard filter屏蔽持有动作时，在同一步释放并调用原评分器选择，不额外等待一帧。
6. 达到2秒自动结束。条件必须先解除、再重新成立才能再次持有；连续低进度不会无间断重新commit。
7. 保持原predict每控制步一次NumPy随机数消耗，避免省略critic时顺带改变模拟器随机流。

0.2米阈值附加1e-12米的浮点边界容差，测试后、冻结前确定，仅处理5.0-4.8的表示误差，不是调参。

**2秒≤0.2米与8秒≤0.8米只共享0.1m/s平均进展阈值，并非逻辑等价。** 在线规则只使用过去；旧8秒规则仍用于离线失败标签，没有偷渡未来数据。

## 3. Preflight：通过，但不是失败专属触发器

只读取此前1024条parent轨迹，不运行新方法、不按新结果选择规则。

| 配置 / case / seed | 旧诊断root | 首次合法触发 | 不晚于旧root |
|---|---:|---:|---|
| 5-square / 81000 / 491 | 7.25s | 3.25s | 是 |
| 10-square / 81001 / 419 | 8.25s | 6.25s | 是 |
| 5-square / 81003 / 419 | 2.00s | 2.00s | 是 |
| 10-square / 81006 / 443 | 11.25s | 3.25s | 是 |

- 743/1024条轨迹存在触发机会，合计29,004个eligible帧。
- 860条原成功轨迹中，591条也存在触发机会，约68.72%。低净目标进展不等于导航故障。
- 40,614个过去窗口净目标进展>0.25m/s的帧没有触发。这由规则直接决定，不是触发器高质量的独立证明。
- 4,126个eligible帧中机器人当前速度>0.5m/s；高速移动也可能绕行或来回移动，不能直接将这些帧称为误触发。
- Preflight只证明机会与合法性，不能证明在更早时刻commit能复现旧oracle-root收益。

## 4. 新测试块与冻结标准

全新86000–86031，2臂×4权重×5/10/20人×circle/square×32case，共1536个完整episode，768个配对。运行前检查本地既有JSON的case/start字段未出现该块；训练协议使用12000起始IL和20000起始RL。没有将80000/81000块再次冒充fresh。

全部在同一RTX3060/PyTorch2.1.0+cu121运行，四个独立进程；用原标量策略、相同case重置、同配对初始世界哈希。不做批量近似控制或远端下载。闭环主块墙钟约279.8秒，不含实现/preflight/测试/归档。

主判据在运行前冻结：square低进度timeout净减少；至少4个不同case获得最小间距≥0.02米的失败→到达；至少2个人数配置净改善；至少3/4种子在square低进度timeout与SR净方向均非负；square碰撞增加≤1pp。

“不能大量破坏原成功”预先量化为每种geometry原成功→失败≤全部384个episode的2pp，即最多7个。circle另外要求SR净下降≤1pp、碰撞净增加≤1pp，各最多3个。这些只是研发门槛，不是显著性检验。

## 5. 全部闭环结果

### 总体

每臂768个episode。SR/CR/TR按原评估器定义，未通过0.02米筛选的到达也完整计入标准SR；筛选只影响“合格挽救”判据。

| 指标 | Parent | Commitment V0 | 差值 |
|---|---:|---:|---:|
| 到达 | 636 | 620 | -16 |
| 成功率 | 82.81% | 80.73% | -2.08pp |
| 碰撞 | 20 | 41 | +21 |
| 碰撞率 | 2.60% | 5.34% | +2.73pp |
| 超时 | 112 | 107 | -5 |
| 超时率 | 14.58% | 13.93% | -0.65pp |
| 到达episode平均用时 | 16.923s | 17.442s | +0.520s |

到达用时两臂样本构成不同，不能单独作公平效率比较。双方均到达的602个配对中，V0平均增加0.616秒。

### 六格结果

每格、每臂128个episode。S/C/T为到达/碰撞/超时；低进度timeout仍使用上轮冻结的8秒离线标签。

| 人数 / geometry | Parent S/C/T | V0 S/C/T | Parent SR | V0 SR | 低进度timeout Parent→V0 |
|---|---|---|---:|---:|---:|
| 5 / circle | 124/0/4 | 122/0/6 | 96.88% | 95.31% | 1→3 |
| 10 / circle | 123/1/4 | 121/4/3 | 96.09% | 94.53% | 4→3 |
| 20 / circle | 118/4/6 | 107/15/6 | 92.19% | 83.59% | 6→6 |
| 5 / square | 96/0/32 | 98/1/29 | 75.00% | 76.56% | 26→23 |
| 10 / square | 99/2/27 | 94/5/29 | 77.34% | 73.44% | 23→25 |
| 20 / square | 76/13/39 | 78/16/34 | 59.38% | 60.94% | 35→30 |

square到达271→270，低进度timeout84→78，碰撞15→22。不能把timeout减少6个单独写成导航改善。

### 原奖励折扣回报

从每条完整episode保存的实际奖励计算，所有成功、碰撞和超时均纳入；不是重新定义的进度分数，也不是oracle最优Q。每格128配对。

| 人数 / geometry | Parent均值 | V0均值 | 配对差值均值 |
|---|---:|---:|---:|
| 5 / circle | 0.524424 | 0.505653 | -0.018771 |
| 10 / circle | 0.496218 | 0.469094 | -0.027124 |
| 20 / circle | 0.398010 | 0.320773 | -0.077237 |
| 5 / square | 0.396184 | 0.394817 | -0.001367 |
| 10 / square | 0.387176 | 0.350228 | -0.036948 |
| 20 / square | 0.212815 | 0.208083 | -0.004732 |

总体768配对：0.402471→0.374774，平均差值-0.027697；square差值-0.014349、circle差值-0.041044。六格原任务回报均值都下降，没有以timeout下降掩盖整体任务代价。

### 救回与破坏

| geometry | Parent失败→到达 | 合格挽救 | 不同合格case ID | Parent到达→碰撞 | Parent到达→超时 | 到达净变化 |
|---|---:|---:|---:|---:|---:|---:|
| square | 14 | 13 | 9 | 5 | 10 | -1 |
| circle | 4 | 4 | 本轮不作为主目标 | 14 | 5 | -15 |

square9个合格case ID：86005、86012、86015、86017、86018、86019、86022、86024、86025。同case不同种子不重复计作独立case。

square还出现4个原timeout→碰撞，circle出现1个；这些转变不能通过只列成功挽救来隐藏。

### 四权重的主目标方向

| seed | square低进度timeout净减少 | square到达净增加 | 两者均非负 |
|---|---:|---:|---|
| 419 | 0 | -3 | 否 |
| 443 | 3 | 3 | 是 |
| 467 | 1 | 0 | 是 |
| 491 | 2 | -1 | 否 |

只有2/4满足要求，不是稳定跨种子增量。

## 6. 冻结判据逐项裁决

| 判据 | 实际结果 | 通过？ |
|---|---|---|
| square低进度timeout净减少 | 84→78，减少6 | 是 |
| 至少4个不同case合格挽救 | 9个case，13个episode | 是 |
| 至少2个人数配置净改善 | 5人、20人；10人恶化 | 是 |
| 至少3/4种子净方向非负 | 2/4 | 否 |
| square碰撞增加≤1pp | +1.82pp | 否 |
| square成功损坏≤2pp | 15/384=3.91% | 否 |
| circle成功损坏≤2pp | 19/384=4.95% | 否 |
| circle SR下降≤1pp | -3.91pp | 否 |
| circle碰撞增加≤1pp | +3.65pp | 否 |

**即便不采用额外量化的成功保护门槛，用户原来的碰撞和3/4种子条件也已经失败。** 不应把本轮否决归咎于额外Gate。

## 7. 一个直接可见的设计缺口

保持原hard filter有一个容易被误解的分支：所有80个候选预测间距均不足0.2米时，Parent关闭硬屏蔽，仍由当前value/risk选动作。V0的“未被屏蔽”在此不代表“安全获准”；旧grid动作可以继续持有，而不再重新比较当前候选。

从已有日志直接数出：

- V0共41次碰撞，其中27次发生在仍持有动作的控制步。
- 这27次的当前候选全部不足0.2米，持有命令预测间距也不足0.2米。
- 141个持有步发生在这种no-safe-candidate状态中。
- 所有持有步都符合本轮继承的hard-mask逻辑；没有“已被该mask屏蔽却继续持有”的实现错误。

这是一处**固定V0设计没有覆盖的安全fallback语义**，不是新的神经模型贡献，也不应把它包装成Parent实现bug。当前证据仅证明风险信号已出现而V0仍持有；尚未做同状态release反事实，不能宣称释放必能挽救27次、解释全部新增碰撞或取得净收益。

全体碰撞还包含14次发生在非持有步，且有双方都会碰撞的case；不能全部算成同一个release机制。

## 8. 触发、时长、间距与成本

- 541/768个V0 episode实际启动过commitment，共816次，最多每episode5次。
- 累计5680个持有控制步、1420秒，平均每episode1.849秒。安全释放150次。
- 双方训练参数量均333,313，新增可学习参数0；新增状态仅9个距离、grid索引、计数和rearm标记。
- 描述性平均predict用时：Parent5.818ms，V06.041ms；持有步1.810ms，V0正常评分步6.440ms。四并发GPU工作进程，非等状态微基准，不能据此宣称部署加速。

| 人数 / geometry | Parent episode最小间距均值 | V0均值 | Parent最差间距 | V0最差间距 |
|---|---:|---:|---:|---:|
| 5 / circle | 0.701689m | 0.735243m | 0.134707m | 0.134707m |
| 10 / circle | 0.343679m | 0.356477m | -0.033390m | -0.182126m |
| 20 / circle | 0.176662m | 0.165464m | -0.072312m | -0.141486m |
| 5 / square | 0.521167m | 0.506830m | 0.124329m | -0.009104m |
| 10 / square | 0.248539m | 0.241932m | -0.032196m | -0.131703m |
| 20 / square | 0.139858m | 0.134534m | -0.080754m | -0.172165m |

平均间距不能代替碰撞率；例如10人circle平均间距增加，但碰撞1→4。

## 9. 已证实、推断、未知

| 等级 | 内容 |
|---|---|
| 已证实 | 旧多步headroom能在部分合法在线干预中形成终局挽救，不只是oracle动作变化 |
| 已证实 | 当前固定规则整体受损，特别是碰撞与circle保护；主裁决B |
| 已证实 | 低进度条件也覆盖大量原成功轨迹；V0有34个原成功→失败 |
| 已证实 | 27个持有期间碰撞在原filter的all-unsafe fallback条件下发生 |
| 有证据支持的推断 | 不能把“无需原mask屏蔽”视为可持续持有的安全许可；一个明确的简单释放对照值得先检验 |
| 尚未证明 | 严格all-unsafe释放能否救回这些碰撞、保护成功且保留挽救 |
| 尚未证明 | 提前释放是否足够，还是触发/动作选择/时长还有独立缺口 |
| 尚未证明 | 自适应commitment是否有效、是否超过强简单修复、是否有方法新颖性 |

## 10. 唯一下一步

**不修改封账V0。若继续，只在新的独立协议下检验“没有任何margin-safe候选时立即交回原策略”的简单释放基线。** 保持2秒/0.2米触发、critic、奖励和其他合同不动，先验证它能否减少真实碰撞并保留挽救，再讨论自适应方法。必须使用新的确认case，不能在已消费的86000块挑规则或checkpoint。

这不是声称该修正能成功，也不是马上追加复杂模型。它来自当前27次具体持有碰撞的合法风险信号，比泛泛设计“智能commit gate”更可检验。若简单修复吸收主要收益，应接受简单方法，不硬造新网络贡献。

## 11. 正确性与资产

- 核对1536份trace哈希、768配对初始世界哈希及规则/四权重/脚本哈希。
- 131,957个控制步的平滑合同最大偏差0。状态机、过去窗口、最多8步、释放及rearm逐步核对。
- 227个未发生commitment的配对，整段实际命令、终局、折扣Q、时长和最小间距完全一致。
- 测试210 passed、3 skipped，68条既有NumPy/SciPy警告；3项旧parity未配置原资产，不冒充通过。
- 合法状态向量沿用原to_array的float32；命令/时间为float64。不能宣称float32状态日志保留全部原float64精度；完整case、动作、checkpoint和源码支持确定性世界回放。
- 首次调度在preflight未完成时请求freeze/run，因缺文件退出；没有执行新case。等待preflight完成后正式冻结，未改规则或覆盖结果。
- 完整1536份episode记录及压缩观测/命令trace保留本地，属于新评估轨迹，不是缺失的原在线RL replay。
- Git包含协议、preflight、逐episode紧凑统计、768配对、校验结果和碰撞上下文；既有checkpoint及完整原始轨迹不公开打包，不声称单独clone可自包含复现。

报告：
/home/abc/workspace/shixu/online-action-commitment-v0.md

完整本地证据：
/home/abc/workspace/shixu/outputs/online-action-commitment-v0/

## 12. 后续单一release确认（2026-10-05）

另一个全新87000–87031块完成Parent/V0/V0.1三臂2304episode；V0.1唯一新增all-unsafe同一步release。相对该块V0，碰撞40→26，原有26个失败挽救全部保留；square相对Parent成功275→288。但square破坏11个原成功、circle破坏9个，circle SR下降1.30pp，原九项Gate仍有三项失败，裁决B。

这提供了release规则的真实增量，不改写本报告86000块的V0负结果；未证明余下损害全部来自trigger。本次单一release验证已完成，不自动追加V0.2或新训练。

完整三臂报告：
/home/abc/workspace/shixu/online-action-commitment-v01.md

<!-- END PRESERVED SOURCE -->


---

<a id="stage-3"></a>

## Source: online-action-commitment-v01.md

Original full-source SHA-256: cf3b6c76e00d3f41feffbaafedf7e5590198905dc09c05cd3bf8fab89da5e1c4

<!-- BEGIN PRESERVED SOURCE -->
# 在线动作持续性V0.1：单一release改动的三臂确认

更新：2026-10-05。完成2304个完整闭环episode。冻结裁决：**B_RESCUE_WITH_UNACCEPTABLE_DAMAGE**。

## 1. 直接结论

**all-unsafe立即释放有真实安全收益，但固定V0.1仍未通过完整保护标准。**

- 同一全新测试块，Parent/V0/V0.1碰撞为24/40/26；release相对V0净减少14次碰撞，没有新增V0原本不碰撞的碰撞episode。
- V0.1保留V0的全部26个失败→成功，并新增12个碰撞→成功和1个超时→成功，同时引入1个V0成功→超时。
- square相对Parent成功275→288，低进度timeout81→74，碰撞14→13；三个人数配置SR均增加，3/4种子主方向非负。
- 但square仍破坏11个Parent原成功，circle破坏9个；circle SR下降1.30pp。三项冻结保护门槛失败，不改Gate宣布通过。
- 总体SR提高1.04pp，但原奖励折扣回报均值略降，双方成功配对的用时增加。不能只报SR就称全面获益。

本轮支持“这个release缺口可以修复一部分V0损害”。**尚未证明剩余损害全部由trigger造成**，持续时长、持有动作选择及其交互仍未分离。没有自动启动V0.2、阈值搜索、自适应网络或训练。

## 2. 唯一控制改动

Parent与V0源文件不改，四个原CV checkpoint不改。V0.1继承V0状态机，只覆盖predict以加入：

> commitment存在时，若当前80个实际平滑候选没有clearance≥原0.2m的动作，在本控制步立即解除commitment，使用Parent按最新合法状态选择的动作。

释放规则也覆盖本步刚启动的commitment。此时Parent已经计算了最新proposal，直接执行它，不重复评分、不持有；已消耗的启动资格仍按原rearm规则处理，连续低进度不能立刻重新commit。这是同一释放约束，不新增另一套触发条件。

其余合同完全沿用：过去2秒进展≤0.2m、距目标>1m；最多8个0.25秒持有步；固定grid index；每步0.3旧命令+0.7 grid；原filter、risk、critic、奖励、tracker、真实history；每步一次原NumPy随机数消耗；安全屏蔽时同一步释放；原episode/reset语义。

Parent的all-unsafe fallback仍照常运行，没有降低安全间距，也没有新挑动作算法。V0.1未训练，无新可学习参数。

## 3. 冻结与测试范围

全新87000–87031，Parent/V0/V0.1 × 四权重419/443/467/491 × 5/10/20人 × circle/square × 每格32case，共2304episode、768个三臂配对。

运行前检查本地既有JSON case/start字段未使用此块，保存源码、原V0脚本、原协议和四权重哈希。独立分支从ac33c4c开始；三臂使用完全相同初始世界，未挑root、未按部分结果停跑。87000块现已消耗，不再称作fresh。

原九项Gate直接调用V0的同一个compare函数，没有重写或放宽。所有候选均在线使用合法观测；隐藏人目标、未来位置、终局标签不进入控制。旧8秒低进度规则仅用于离线失败分类。

同一RTX3060，PyTorch2.1.0+cu121，四进程、CPU线程1。主闭环墙钟约394.1秒，不含实现、测试和归档。没有远端下载或新训练。

## 4. 总体结果

每臂768episode；所有到达均计入标准SR，0.02m仅用于冻结的合格挽救筛选。

| 指标 | Parent | V0 | V0.1 |
|---|---:|---:|---:|
| 到达 | 640 | 636 | 648 |
| 成功率 | 83.33% | 82.81% | 84.38% |
| 碰撞 | 24 | 40 | 26 |
| 碰撞率 | 3.13% | 5.21% | 3.39% |
| 超时 | 104 | 92 | 94 |
| 超时率 | 13.54% | 11.98% | 12.24% |
| 原奖励折扣回报均值 | 0.402961 | 0.386603 | 0.400351 |
| 到达episode平均用时 | 16.993s | 17.522s | 17.517s |

到达用时的样本构成不同，不单独作公平效率结论。Parent/V0.1共同成功的620个配对中，V0.1平均增加0.467秒；square264个共同成功配对增加0.315秒，circle356个增加0.580秒。

### 六格完整终局

每格每臂128episode。S/C/T为到达/碰撞/超时。

| 人数 / geometry | Parent S/C/T | V0 S/C/T | V0.1 S/C/T | 低进度timeout Parent/V0/V0.1 |
|---|---|---|---|---|
| 5 / circle | 126/0/2 | 125/0/3 | 125/0/3 | 1/2/2 |
| 10 / circle | 125/0/3 | 124/0/4 | 123/0/5 | 2/3/4 |
| 20 / circle | 114/10/4 | 106/19/3 | 112/13/3 | 4/3/3 |
| 5 / square | 93/1/34 | 98/1/29 | 98/1/29 | 24/24/24 |
| 10 / square | 96/1/31 | 97/4/27 | 100/1/27 | 27/23/23 |
| 20 / square | 86/12/30 | 86/16/26 | 90/11/27 | 30/26/27 |

### Geometry及原任务回报

| 项目 | Parent | V0 | V0.1 |
|---|---:|---:|---:|
| square到达 | 275 | 281 | 288 |
| square碰撞 | 14 | 21 | 13 |
| square超时 | 95 | 82 | 83 |
| square低进度timeout | 81 | 73 | 74 |
| square折扣回报均值 | 0.333370 | 0.330874 | 0.347679 |
| circle到达 | 365 | 355 | 360 |
| circle碰撞 | 10 | 19 | 13 |
| circle超时 | 9 | 10 | 11 |
| circle折扣回报均值 | 0.472552 | 0.442332 | 0.453022 |

square SR71.61%→75.00%，+3.39pp；circle95.05%→93.75%，-1.30pp。V0.1原任务回报相对Parent：square+0.014309，circle-0.019530，总体-0.002611。不能将square收益扩大成所有场景全面改善。

## 5. Release究竟修复了什么

本节全部比较同一87000块的V0/V0.1，不能把上轮86000块的27次碰撞与本轮相减。

| 对应指标 | V0 | V0.1 |
|---|---:|---:|
| 总碰撞 | 40 | 26 |
| 持有期间all-unsafe碰撞 | 23 | 0 |
| 持有期间all-unsafe控制步 | 87 | 0 |
| all-unsafe释放次数 | 0 | 42 |

0个持有/all-unsafe碰撞由规则保证，不能单独当安全改进。实际终局提供额外证据：

- V0的23个持有/all-unsafe碰撞，在V0.1中12个到达、2个超时、9个仍碰撞。真正消除碰撞14个，不是23个全部救回。
- V0其余17个碰撞在V0.1中仍然碰撞。不能声称release解决了全部安全问题。
- 三臂完整续跑中，V0全部40个碰撞变成12到达、2超时、26碰撞；没有新增V0原本非碰撞的碰撞。
- V0已有26个Parent失败→到达全部保留，包含square23个、circle3个。
- 另有1个V0超时→到达、1个V0到达→超时。release不是完全没有副作用。

42次新release发生在40个episode。728个没有新release的V0/V0.1配对，完整实际命令、终局、回报、到达时间和最小间距完全一致；其余40个首次release前的命令前缀也完全一致。这将差分限定到新release规则，而非模型、起点或其他控制合同变化。

## 6. 成功保护与四种子

| geometry | V0失败→到达 | V0破坏Parent成功 | V0.1失败→到达 | V0.1合格挽救 | V0.1破坏Parent成功 |
|---|---:|---:|---:|---:|---:|
| square | 23 | 17 | 24 | 24 | 11 |
| circle | 3 | 13 | 4 | 2 | 9 |

V0.1的square11个损害全部为Parent成功→超时，没有成功→碰撞；circle损害为6个成功→碰撞、3个成功→超时。不能因为总体SR提高而省略这20个损害。

square24个合格挽救来自16个不同case ID：87000、87002、87004、87006、87008、87010、87011、87012、87013、87015、87016、87017、87018、87020、87023、87031。不同权重同case不重复计作独立case。

| seed | square低进度timeout净减少 | square到达净增加 | 两者均非负 |
|---|---:|---:|---|
| 419 | 7 | 9 | 是 |
| 443 | 1 | 3 | 是 |
| 467 | 4 | 6 | 是 |
| 491 | -5 | -5 | 否 |

3/4通过种子方向门槛，但491仍恶化，不能称所有种子稳定改善。5/10/20人square到达净增加5/4/4；低进度timeout净减少0/4/3，因此主目标上的两个人数配置是10和20人。

## 7. 原九项Gate：不改结论

| 原冻结标准 | V0.1实际结果 | 通过？ |
|---|---|---|
| square低进度timeout净减少 | 81→74，减少7 | 是 |
| ≥4不同case合格挽救 | 16个case、24个episode | 是 |
| ≥2人数配置主目标净改善 | 10人、20人 | 是 |
| ≥3/4种子两项方向非负 | 3/4 | 是 |
| square碰撞增加≤1pp | 减少1个，-0.26pp | 是 |
| square成功破坏≤2pp，即≤7个 | 11/384=2.86% | 否 |
| circle成功破坏≤2pp，即≤7个 | 9/384=2.34% | 否 |
| circle SR净损失≤1pp，即≤3个 | 少5个，-1.30pp | 否 |
| circle碰撞增加≤1pp，即≤3个 | 多3个，+0.78pp | 是 |

**6/9通过、3/9失败，冻结裁决B，不宣布METHOD_ENTRY。** 本轮有真实机制增量，与“完全无效”不同；也不能因差几个episode就修改保护标准。

## 8. 间距与计算成本

| 人数 / geometry | Parent最小间距均值 | V0均值 | V0.1均值 | 三臂最差间距 Parent/V0/V0.1 |
|---|---:|---:|---:|---|
| 5 / circle | 0.693588m | 0.722497m | 0.722497m | 0.184333/0.180897/0.180897m |
| 10 / circle | 0.330818m | 0.359631m | 0.360342m | 0.088574/0.078884/0.141728m |
| 20 / circle | 0.169492m | 0.155790m | 0.168303m | -0.122327/-0.210874/-0.117076m |
| 5 / square | 0.549432m | 0.551748m | 0.551748m | -0.010018/-0.010018/-0.010018m |
| 10 / square | 0.268564m | 0.270414m | 0.277525m | -0.009158/-0.212973/-0.009158m |
| 20 / square | 0.142657m | 0.136232m | 0.146188m | -0.141855/-0.437246/-0.141855m |

均值不替代碰撞计数。三臂模型参数量均333,313；新可学习参数0。V0/V0.1累计持有1431/1391秒，启动816/819次；实际持有过的episode543/532。包含本步启动即release的请求时，启动计数不等于完整2秒持有。

描述性平均predict延迟：Parent5.559ms、V05.788ms、V0.1 5.790ms。四并发GPU进程，非等状态部署微基准，不宣称加速或统计等效。

## 9. 已证实 / 推断 / 未知

| 等级 | 内容 |
|---|---|
| 已证实 | 单一all-unsafe释放在该fresh块相对V0减少14次真实碰撞，保留全部旧挽救 |
| 已证实 | V0.1在square相对Parent具有成功、timeout及原任务回报增量 |
| 已证实 | V0.1仍破坏20个Parent成功，circle与原保护门槛不合格 |
| 已证实 | 低进度条件覆盖原成功轨迹，不是失败专属信号；但成功时曾满足条件不自动等于误触发 |
| 有证据支持的推断 | fixed commitment的继续/释放语义具有实质影响，不能只按原hard-mask判断“允许持有” |
| 有证据支持的推断 | 是否启动持有值得成为下一项独立受控问题，而不是继续修all-unsafe分支 |
| 尚未证明 | 剩余损害主要或全部来自trigger过宽，而非时长、持有动作及其交互 |
| 尚未证明 | 仅改变启动规则能通过保护标准，或需要自适应时长/序列评价 |
| 尚未证明 | 该简单方法具有未被先例覆盖的方法新颖性，或足以形成论文结论 |

## 10. 唯一下一步与停止边界

**保留V0.1为经过实测的强简单baseline，不再追加V0.2/V0.3释放补丁或时长/阈值扫描。** 若继续，下一项限定为验证“合法在线启动信号能否保留挽救、避免破坏正常绕行”的受控实验；持续时长和动作选择必须作为尚未排除的替代解释保留，不能先宣布trigger就是根因。

这一下一步需要单独冻结协议和新确认块。本轮不自动开发adaptive gate、神经网络、critic、新reward或KDA，不因局部正信号直接宣布正式方法成功。旧4root多步headroom以及V0负结果全部保留。

## 11. 正确性、校验细节与复现资产

- 完整2304条trace哈希、768个三臂初始世界、四权重和冻结源码核对通过。
- 194,554个控制步的平滑递推最大偏差0；过去窗口、同一步release、最多8步和rearm状态机核对通过。合法观测入口继承原observer，控制代码未读取隐藏目标或未来真值。
- Parent/V0未实际持有的225配对、Parent/V0.1未实际持有的236配对完整轨迹一致；V0/V0.1无新release的728配对一致，40个release配对首次release前命令一致。
- 全仓库214 passed、3 skipped、68条既有NumPy/SciPy警告。3项旧parity未配置原资产，不冒充通过。
- 原冻结脚本的validate按文件名字典序聚合，导致V0总体mean_q重算差1.1e-16；只有这一均值不同，按原数值seed/population/case顺序全部一致。新增独立校验脚本统一顺序，记录哈希；没有修改冻结控制脚本、协议、episode或summary，没有重跑。
- 原始输入状态使用native to_array float32，实际动作和时间float64；不宣称状态日志完整保留原float64精度。
- Git保存协议、2304条紧凑episode统计、两组768配对、release对照和校验记录。完整观测/动作trace与既有checkpoint本地保存，不宣称单独clone可自包含复现。

运行入口：python -m experiments.online_action_commitment_v01 run；独立校验：python -m experiments.validate_commitment_v01。冻结文件不覆盖，已完成块不作为新的确认数据重复利用。

报告：
/home/abc/workspace/shixu/online-action-commitment-v01.md

完整本地证据：
/home/abc/workspace/shixu/outputs/online-action-commitment-v01/

<!-- END PRESERVED SOURCE -->


---

<a id="stage-4"></a>

## Source: commit-advantage-learnability.md

Original full-source SHA-256: 905449272371cc33ea1e6b750a5f7e343a010e93f051118431d96ecaf73f6f05

<!-- BEGIN PRESERVED SOURCE -->
# Commit-vs-Replan：可学习性与四臂闭环结果

更新：2026-10-05。已完成384条五人标签轨迹、1,259个配对root、3,072条fresh闭环。

**冻结裁决：LEARNED_INITIATION_NOT_CONFIRMED。不是完全无收益，也不是METHOD_ENTRY_FOUND。**

## 1. 结论先说清

- 线性gate通过预先冻结的离线开发准入；MLP没有通过跨checkpoint留出。这个准入不是方法成功。
- fresh完整闭环中，MLP相对Parent到达677→688、碰撞23→20，原折扣回报均值0.434260→0.441807。确有受控的正向结果。
- square低进度timeout54→48、到达312→317，但破坏8个原成功；冻结上限是7个。因此MLP只通过8/9项Parent对照门槛，不能因只差一个就修改标准。
- 20人square：Parent到达91，MLP90；该格原回报也下降。尚未建立所要求的密集场景泛化优势。
- 线性gate也通过8/9项，但破坏19个square原成功，而且20人square明显恶化。不能用整体SR上升掩盖这一点。
- 固定V0.1在本块square没有复现上块的净改善。87000块的正证据保留，不能把它扩大成稳定跨case结论，也不能用本块去改写旧数字。

本轮没有调阈值、换checkpoint、改duration、重训critic或启动新的gate版本。结果封账，停止本次自动研发。

## 2. 这次真正实现了什么

Parent还是原CV后继模型、actor价值网络、80个原生动作、原reward/filter/risk/history和平滑合同。四个原权重419/443/467/491均为原五人circle IL50/在线MC3000资产，本轮没有重新执行IL或RL。

新增部分只决定是否启动最多2秒的持有：

> 当前Parent选动作 → 估计一次commit相对Parent续跑的回报差 → 大于0时允许启动 → V0.1安全释放 → 恢复Parent。

不是新导航动作策略，不是新critic，不学习duration或release。线性模型19参数；MLP是146→16→1，共2,369参数。四个Parent共用同一个case-only线性模型和同一个case-only MLP，**不是四个独立训练的gate**。

沿用每步一次0.3旧命令+0.7固定grid的平滑；每步0.25秒更新合法观测、tracker和真实history。持有动作被原margin检查挡住、或80动作全部没有clearance≥0.2m时，同一步release并执行最新Parent动作。原Parent的all-unsafe fallback不变。

启动后最多8步；资格条件清除再出现才可再次启动。不在安全持有期间重新学习release或重算gate。部署不使用未来、隐藏人目标、内部ORCA状态、case/seed/geometry编号。

## 3. 标签合同与数据

只在原生五人circle采集。square及10/20人没有参与拟合或模型选择，保持原训练域并测试OOD。这是本次设定，不是声称五人训练必须只用circle。

96个新case90000–90095 × 四个冻结Parent，384条完整轨迹。选root不看方法结果或Parent终局：固定2/6/10秒，加首次“过去2秒进度≤0.2m”的合法时刻；去重、距自身目标>1m。

每个root恢复相同float64世界、合法history、tracker和NumPy随机状态：

- A：原Parent一直续跑到终局。
- B：当前Parent grid最多持有8步，沿用V0.1释放，之后**永久Parent**。
- 标签y=G(B)−G(A)，使用原gamma=0.99及原环境奖励，从root开始折扣。

这是指定Parent continuation下的一次控制干预差值，**不是最优advantage，也不是反复启动gate的Q函数**。模拟器隐状态仅用于恢复世界与生成离线标签，不进入gate输入。

384条Parent之外另有2,518条恢复root的续跑分支，不能将它们计成独立场景样本。Parent/V0.1没有使用新增监督标签；线性与MLP获得相同标签。本轮单列了额外数据/仿真成本，尚未做等额新增交互预算重训Parent的论文级比较。

三个89999预检root不进入数据集。全部1,259个正式root的A分支与原Parent剩余轨迹逐动作一致，终局/折扣回报一致；384个输入归档哈希回查通过。

| 划分 | case | Parent轨迹数 | root数 | y>0.005 | y<−0.005 | 其余 |
|---|---|---:|---:|---:|---:|---:|
| 训练 | 90000–90063 | 256 | 840 | 197 | 449 | 194 |
| 验证 | 90064–90079 | 64 | 207 | 57 | 105 | 45 |
| 测试 | 90080–90095 | 64 | 212 | 56 | 123 | 33 |

同case的全部root不跨划分。每个(seed,case)总权重相同，防止root多的轨迹主导损失。四次leave-checkpoint-out同时保留未见seed和未见case；另做同checkpoint、未见case的case-only检验。测试标签不用于refit。

训练840个root中，commit只产生3个timeout→success，却有16个success→timeout；验证分别1个和2个。标签域没有碰撞分支，不能声称训练时学到了全面安全收益判别。训练标签以正常成功轨迹为主，是重要限制。

## 4. 输入及训练没有变复杂

18个合法标量：自身目标距离、过去2秒进度、窗口完整性、当前grid/平滑命令、所选score、top1-top2 gap、预测clearance、安全候选比例、近期grid变化、机器人速度/命令变化、可见比例、track年龄、当前间距、目标方向一致性。

MLP另读取Parent已计算的128维“所选候选在原value head之前”的特征。它是合法历史及CV假想后继的模型表征，**不是额外真值、不是单纯current embedding**。线性模型只用18个标量。

线性ridge固定alpha=1；MLP固定16宽度、Adam0.001、200次full-batch更新、按验证MSE保存。仅训练集标准化，未搜索特征/超参。case-only MLP选中的epoch为42。

不同Parent的潜特征坐标未证明对齐。MLP跨checkpoint失败不能唯一归因于“advantage不可学习”；语义线性模型是避免这种误判的独立对照。

## 5. 离线准入：弱信号必须按弱信号报告

表中收益是episode平衡的“一次被gate选中干预的真实回报差”均值，不是线上SR。

| 模型 | 未见case收益 | 同时未见case/seed收益 | 跨seed收益非负 | 跨seed平衡准确率 | 离线准入 |
|---|---:|---:|---:|---:|---|
| 语义线性 | +0.003781 | +0.001206 | 3/4 | 51.69% | 通过冻结开发门槛 |
| 小MLP | +0.003163 | −0.000536 | 2/4 | 48.61% | 未通过 |

跨seed线性收益419/443/467/491：+0.005278/+0.001162/−0.001931/+0.000316。MLP：−0.000133/+0.000553/−0.002564/0。

线性模型的未见case收益去掉90093后变成−0.000091，显示正收益集中。这个补充检查不是事后改Gate，而是说明信号强度；不能宣称已可靠识别收益。原冻结准入是任一模型case-only及联合留出收益>0.001、至少3/4 seed非负、正负标签各覆盖至少4个测试case、额外碰撞≤1pp。线性通过，因此继续完整闭环，不仅挑它跑，也保留MLP作比较。

## 6. Fresh完整闭环

最初拟用91000块，冻结前查到旧PaS使用记录；没有运行新方法就弃用。正式固定99000–99031，本地既有JSON case/start字段未见使用。

Parent/V0.1/线性/MLP × 四个Parent权重 × 5/10/20人 × circle/square × 每格32case，共3,072条、768组四臂相同初始世界。没有结果导向停跑，全部跑到原生终局。99000块现已消耗，不能再称fresh。

### 总体结果：每臂768条

| 指标 | Parent | 固定V0.1 | 语义线性gate | MLP gate |
|---|---:|---:|---:|---:|
| 到达 | 677 | 671 | 684 | 688 |
| SR | 88.15% | 87.37% | 89.06% | 89.58% |
| 碰撞 | 23 | 22 | 22 | 20 |
| 超时 | 68 | 75 | 62 | 60 |
| 原折扣回报均值 | 0.434260 | 0.418251 | 0.440631 | 0.441807 |
| 成功episode平均时间 | 16.763s | 17.493s | 16.648s | 16.863s |
| 全episode最小间距的均值 | 0.363225m | 0.365928m | 0.333193m | 0.345552m |
| 全块最差间距 | −0.090052m | −0.090052m | −0.138470m | −0.090390m |

成功用时的样本构成不同，不直接当效率优势。MLP与Parent共同成功的359个circle配对平均−0.019s；304个square配对平均+0.227s。

### 六格终局：每格每臂128条，S/C/T为成功/碰撞/超时

| 场景 | Parent S/C/T | V0.1 S/C/T | 线性 S/C/T | MLP S/C/T |
|---|---|---|---|---|
| 5 circle | 123/0/5 | 123/0/5 | 126/0/2 | 124/0/4 |
| 10 circle | 125/1/2 | 121/1/6 | 124/0/4 | 127/0/1 |
| 20 circle | 117/6/5 | 116/5/7 | 121/3/4 | 120/3/5 |
| 5 square | 109/0/19 | 110/0/18 | 117/0/11 | 112/0/16 |
| 10 square | 112/1/15 | 114/1/13 | 113/3/12 | 115/1/12 |
| 20 square | 91/15/22 | 87/15/26 | 83/16/29 | 90/16/22 |

MLP在5/10人square各净增3个成功，20人净少1个。20人square原Q：Parent0.265783、线性0.223533、MLP0.249994。**不能写成5→10→20泛化全部成立。**

### Square主目标及circle保护

| 指标 | Parent | V0.1 | 线性 | MLP |
|---|---:|---:|---:|---:|
| square成功 | 312 | 311 | 313 | 317 |
| square低进度timeout | 54 | 55 | 50 | 48 |
| square碰撞 | 16 | 16 | 19 | 17 |
| square Q均值 | 0.397277 | 0.381582 | 0.393710 | 0.400264 |
| circle成功 | 365 | 360 | 371 | 371 |
| circle碰撞 | 7 | 6 | 3 | 3 |
| circle Q均值 | 0.471244 | 0.454920 | 0.487553 | 0.483351 |
| square失败→成功 | — | 11 | 20 | 13 |
| square成功→失败 | — | 12 | 19 | 8 |
| circle失败→成功 | — | 4 | 12 | 12 |
| circle成功→失败 | — | 9 | 6 | 6 |

MLP square的8个损害为7个成功→timeout、1个成功→collision；circle6个为4个timeout、2个collision。MLP square13个挽救均满足原0.02m全episode间距筛选，来自11个独立case编号。circle12个挽救中11个满足该筛选；不将不满足者称作合格安全挽救。

MLP只比线性多4个总体成功，总体Q多0.001176；尚未证明非线性/潜特征具有稳定额外收益。线性的circle Q反而更高。

## 7. 原九项门槛，不改口径

| 原冻结标准 | 线性 | MLP |
|---|---|---|
| square低进度timeout净减少 | 减少4，通过 | 减少6，通过 |
| ≥4独立case合格挽救 | 15个case，通过 | 11个case，通过 |
| ≥2人数配置SR与主目标同时净改善 | 5/10人，通过 | 5/10人，通过 |
| ≥3/4 Parent seed主方向非负 | 3/4，通过 | 3/4，通过 |
| square碰撞增加≤1pp | +3/384，通过 | +1/384，通过 |
| square原成功破坏≤2pp，即≤7个 | 19/384，失败 | 8/384，失败 |
| circle原成功破坏≤2pp，即≤7个 | 6/384，通过 | 6/384，通过 |
| circle SR净损失≤1pp | 净增6，通过 | 净增6，通过 |
| circle碰撞增加≤1pp | 净少4，通过 | 净少4，通过 |

MLP还在square SR/Q上超过固定V0.1与线性；但这一增量条件不能替代失败的原保护标准。两个学习臂均8/9，不升级METHOD_ENTRY。

| seed | 线性square成功净增 / 低timeout净减 | MLP对应指标 |
|---|---:|---:|
| 419 | +2 / +4 | +3 / +3 |
| 443 | −4 / −3 | −1 / +1 |
| 467 | +1 / 0 | +1 / 0 |
| 491 | +2 / +3 | +2 / +2 |

四行是四个冻结Parent训练seed的评估，不是四个gate训练seed实验。不能拿3/4的方向计数充当统计显著性。

## 8. 正确性与成本

224 tests通过，3个旧legacy路径相关测试跳过。3072条轨迹逐条校验原折扣奖励、0.25秒时钟、实际grid的单次平滑、同一步release及保存输入到gate输出的数值一致性。

实际控制步239,772。没有持有blocked/all-unsafe动作。未发生持有的Parent配对整条轨迹相同：V0.1 265组、线性11组、MLP42组。大多数episode都会启动至少一次学习gate，不应把它宣传为只针对少数失败的稀疏接管。

标签采集墙钟201.1秒；3072条闭环493.3秒。本地RTX3060、四进程、CPU线程1，未向4090下载资产。

| 成本 | 线性 | MLP |
|---|---:|---:|
| 新参数 | 19 | 2369 |
| gate文件 | 1924 B | 15424 B |
| 仅gate CPU调用均值，1000次 | 0.0145ms | 0.0347ms |
| 原样确定性拟合复算计时 | 0.0104s | 0.5355s |
| fresh并发完整predict均值 | 5.796ms | 6.252ms |

原十次fit未逐模型计时；表中拟合时间是只用原train/val的确定性权重复算，参数与原文件逐项完全相同，**没有重选checkpoint、没有替换结果**。不能把它冒称原作业精确计时。

同一冻结root/state/history下，100次轮换CUDA同步计时的score路径：Parent2.795ms、线性score+特征+gate3.441ms、MLP3.468ms。增加约0.65–0.67ms，主要路径不是仅0.035ms的MLP调用。该测量排除额外wrapper预检查/历史写入，不是完整机器人部署延迟。

完整predict均值还受并发与轨迹构成影响：Parent5.429ms、V0.1 5.694ms。因此现在只能报告代价，不声称成本已被收益充分证明值得。

## 9. 已证实、推断、未证明

| 等级 | 判断 |
|---|---|
| 已证实 | 固定五人监督训练的两个gate在线改变了commit启动；有实际终局和原奖励差分，不只是动作敏感性 |
| 已证实 | MLP在本fresh块总体、square主指标上有正增量，同时保留8个square成功损害和20人square退化 |
| 已证实 | 线性是很强的简单替代；MLP没有取得通过冻结保护标准的完整优势 |
| 已证实 | MLP未通过离线seed留出，却在线有正结果；弱离线probe不能当整条机制的终局判决 |
| 有证据支持的推断 | 使用commit相对replan的收益监督，比固定低进度trigger更值得保留为研究机制 |
| 尚未证明 | 原策略失败唯一由高频replanning造成；剩余损害唯一来自启动，而非时长/动作/重复启动交互 |
| 尚未证明 | 一次Parent-tail标签与重复部署的差分就是当前损害的原因；它们合同确有差别，但尚无因果消融 |
| 尚未证明 | 非线性gate/潜特征比19参数语义线性具有可重复、跨gate训练seed的优势 |
| 尚未证明 | 20人square泛化、完整安全保护和成本收益足以支撑SCI主张 |

## 10. 新颖性边界与唯一下一步

[AFST原论文](https://arxiv.org/abs/2108.06161)已将自适应执行时间纳入机器人导航SMDP。一般options也已研究启动/终止；本轮没有完成exact-prior审计，不能把“按advantage启动持有”直接叫新算法，更不能声称首次自适应动作持续时间。

**建议的唯一下一步是：不改这两个gate、duration或阈值，先做一块独立fresh四臂确认，沿用全部保护门槛并保留20人square这一已暴露薄弱格。** 本轮不自动执行。

理由：现在有真实闭环增量，但MLP保护失败、线性替代强、跨checkpoint离线信号弱。独立确认能区分“值得继续的方法信号”与“case块上的开发正结果”；继续加结构或调trigger反而会破坏可解释性。

若确认仍有square原成功损害超标或密集square退化，就不能宣称合格方法；届时应明确诊断一个具体机制再决定一次有依据的修改，不能连续试阈值直到赢。若确认合格，再做gate训练seed、潜特征/语义输入和强简单方法消融，并精确核查先例。现在不提前实施这些分支。

## 11. 归档与源码

父体提交e2f0621；标签/学习阶段源码及冻结数据提交ed6be93，独立分支research/commit-advantage-learnability。label阶段源码哈希保留；新增在线module不改旧source字节，在线协议通过排除新module回查原label source，再单独冻结完整新source。

保留384个本地输入snapshot、各root两分支续跑命令、3072个完整trace及episode元数据。Git只上传源码、一个本报告、gate权重、标签特征/回报及精简冻结结果，不上传全部snapshot/trace。隐藏真值恢复文件不属于部署输入。

主证据：

/home/abc/workspace/shixu/outputs/commit-advantage-probe/protocol.json

/home/abc/workspace/shixu/outputs/commit-advantage-probe/dataset.json

/home/abc/workspace/shixu/outputs/commit-advantage-probe/learnability.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/protocol.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/summary.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/validation.json

/home/abc/workspace/shixu/outputs/online-commit-advantage/runtime-profile.json

本地只读复核：PYTHONPATH=vendor:. /usr/bin/python -m experiments.online_commit_advantage validate。重新采集标签应使用ed6be93的干净源码与原critic资产；不能在已保存probe上覆盖训练，也不能把已使用case重新称作独立测试。

<!-- END PRESERVED SOURCE -->


---

<a id="stage-5"></a>

## Source: AUTONAV_OVERNIGHT_REPORT.md

Original full-source SHA-256: 9a990179029cf1c231d24728d061e22ac9fff8dfc1af18b4c7dca6f391cc4f8d

<!-- BEGIN PRESERVED SOURCE -->
# AUTONAV OVERNIGHT REPORT

日期：2026-10-06。最终判定：FINAL_CONFIRMATION_FAILED。

11/60个offline实验，8/8个dev1候选，1/3个dev2候选，唯一一次2304-episode final已完成。final失败后没有修改方法、重训、选新checkpoint或重新确认。负结果全部保留。

## 1. 最佳开发方法与冻结范围

最佳开发候选：mlp-physical。它是唯一同时通过两个预先冻结development门槛的版本，不是最终合格方法。

结构：10个合法物理/动作连续性特征 → Linear(10,16) → ReLU → Linear(16,1)。193个可训练参数；只预测commit advantage的符号，不生成导航动作。

使用的10项：自身目标距离、Parent所选grid速度、平滑命令与目标方向的一致性、两秒CV hold最小间距、安全候选比例、当前相对运动TTC、过去动作索引切换比例、过去命令变化、当前命令与上一步的航向变化、两秒CV hold目标进度。没有case ID、人数标签、场景标签、行人目标、ORCA内部状态或真实未来输入。

训练：episode-balanced原始delta-Q MSE；train-only标准化；同一固定随机种子616、300 epochs、Adam lr=0.003、weight decay=0.001。不用dev选epoch。target只作正比例缩放，部署输出>0才接受proposal；没有拟合accept阈值。

范围：
- 只在原V0.1低进度proposal出现时accept/reject；拒绝也消耗当次启动机会，须原低进度条件清除后才能重新arm。
- 所选grid、0.3/0.7平滑递推、最多2秒、每0.25秒更新、原安全检查与all-unsafe release均未变化。
- Parent四checkpoint、critic、reward、observer、history、动作支持及导航训练协议全程冻结。没有新增IL/RL导航训练。
- prepare/features/evaluate及相关核心文件共43个哈希绑定，冻结后仅method.py发生科研改动；报告与结果归档不改变科学实现。

新标签：130000–130511五人circle，512 case × 4 checkpoint = 2048条Parent轨迹，采集每个合法proposal，得到1320个配对root。每root恢复同一世界/history/RNG，分别Parent续跑和一次V0.1 commit后永久Parent续跑，直至终止。全部1320个Parent分支通过原轨迹命令/终局/折扣return恢复一致性检查。

train：130000–130383，971 root，18个挽救标签、20个损害标签。offline dev：130384–130511，349 root，21个挽救标签、5个损害标签。跨checkpoint同case从未跨split。offline dev的21个挽救标签仅来自4个case，不能解释成21个独立case的证据。

五人circle数据未出现新增碰撞标签，故不能从训练集证明gate学会了密集场景碰撞辨别。隐藏状态只保存在离线恢复资产中，不进入部署feature。

方法实验commit：4f9df4063075daec84debc9cd416895dd543983f。
工作文件原样晋级commit：7c2167f；与原实验归档代码逐字节一致。
method SHA256：ad82edc2c8dd3ab8560ecf764cbfa97a7667315d0c899aafd57743c283aa2c56。
weights SHA256：29277196ce5ee0397c3bbe52d9d59c8015cd0614f89f1581d2815acbd61b65d4。
protocol SHA256：e43cb74a0daaa9e28309bf5330e05cccbeeec6712926446cc47991e604debc57。
dataset SHA256：d733e879f19f6013d4f15108185dd7b0e53a9f90bb9268bf67280f0adfe0dd43。

## 2. 完整实验树与Keep/Discard

每条边是一个机制测试或简化，不是阈值扫描。共有11个独立、评估前已commit的实验；晋级时没有再fit。全接受基准没有作为新的闭环候选重复运行。

```text
rule-accept-all
  |- rule-forward
  |    |- rule-forward-cv
  |    |- linear-advantage
  |         |- linear-physical
  |         |- logistic-utility
  |              |- mlp-utility
  |                   |- mlp-advantage
  |                        |- mlp-physical
  |                             |- mlp-physical-utility
  |- rule-cv-clearance
```

- rule-accept-all：基准，全接受；平均advantage为负，未减损害。
- rule-forward：两秒CV目标进度为正；dev1只保留11/24个square挽救。
- rule-cv-clearance：两秒CV间距≥0.2m；平均advantage为负。
- rule-forward-cv：进度与CV间距同时满足；dev1挽救不足。
- linear-advantage：24维线性delta-Q回归；dev1挽救不足。
- linear-physical：10维物理特征线性回归；dev1挽救不足。
- logistic-utility：按|delta-Q|加权分类；dev1挽救不足。
- mlp-utility：24维、16宽、加权分类；dev1 square挽救11/24。
- mlp-advantage：同架构改为delta-Q MSE；offline advantage为负。
- mlp-physical：10维、16宽、delta-Q MSE；两开发块通过，final失败。
- mlp-physical-utility：同10维架构改为加权分类；dev1挽救不足。

Offline表：advantage是episode-balanced E[accept × 原delta-Q]，拒绝计零；不是总体导航回报。coverage也是episode-balanced。挽救/损害为root标签计数，不是整episode SR。所有offline新增碰撞标签计数均为0，不能视为泛化安全证明。

| 方法 | Commit | 参数 | Fit秒 | Advantage | Accept覆盖 | 正收益覆盖 | 挽救/损害root | Gate读取µs | Offline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| rule-accept-all | 2702168 | 0 | 0.0000 | -0.017585 | 100.0% | 100.0% | 21/5 | 1.03 | DISCARD |
| rule-forward | f22aa7e | 0 | 0.0000 | 0.008016 | 64.1% | 90.0% | 21/0 | 1.12 | KEEP |
| rule-cv-clearance | ec7f6f1 | 0 | 0.0000 | -0.022018 | 92.7% | 87.6% | 5/5 | 1.14 | DISCARD |
| rule-forward-cv | 378cae7 | 0 | 0.0000 | 0.003111 | 60.0% | 77.6% | 5/0 | 1.23 | KEEP |
| linear-advantage | 313a8e3 | 25 | 0.0043 | 0.011039 | 22.7% | 43.7% | 21/0 | 5.78 | KEEP |
| linear-physical | e39fbb6 | 11 | 0.0019 | 0.010353 | 24.5% | 46.6% | 21/0 | 7.15 | KEEP |
| logistic-utility | a57a97a | 25 | 0.4903 | 0.010109 | 29.9% | 53.0% | 20/0 | 4.52 | KEEP |
| mlp-utility | b629361 | 417 | 0.5385 | 0.010992 | 26.9% | 49.1% | 21/0 | 7.64 | KEEP |
| mlp-advantage | 59b9324 | 417 | 0.5197 | -0.001987 | 34.9% | 34.5% | 20/2 | 7.49 | DISCARD |
| mlp-physical | 4f9df40 | 193 | 0.5295 | 0.006465 | 27.0% | 43.2% | 20/0 | 8.76 | KEEP |
| mlp-physical-utility | 02e10f1 | 193 | 0.5393 | 0.006196 | 24.2% | 40.3% | 20/0 | 8.86 | KEEP |

Offline门槛在数据生成前冻结：正收益/挽救标签需有支持，accept覆盖≥5%，正收益覆盖≥10%，挽救权重保留≥25%，平均干预advantage>0，损害较全接受降低≥25%，新增碰撞受限。没有never-commit版本被当成成功。

Dev门槛同样预先冻结：circle与square损害各降低≥25%；保留≥50% square及总V0.1挽救数；≥4个合格square挽救case；square低进度timeout不增加；各几何新增碰撞≤1pp。这是研发筛选，不是统计显著性检验。final则只使用原九项Gate，未加入或放松新门槛。

Dev1使用旧87000–87031块的已验证Parent/V0.1缓存，候选新跑768 episodes。每候选同场景、同权重、同初始世界；没有重跑缓存Parent/V0.1。S=success，C=collision，T=timeout；下表净收益/损害均相对Parent。

| 方法 | 挽救square/circle | 损害square/circle | 成功净增square/circle | Square低timeout净减 | 原九Gate | Dev1裁决 |
|---|---:|---:|---:|---:|---:|---|
| rule-forward | 11/4 | 7/4 | 4/0 | 4 | 9/9 | DISCARD：挽救保留不足 |
| rule-forward-cv | 9/2 | 6/3 | 3/-1 | 3 | 8/9 | DISCARD：挽救保留不足 |
| linear-advantage | 7/2 | 4/3 | 3/-1 | 3 | 9/9 | DISCARD：挽救保留不足 |
| linear-physical | 7/1 | 3/3 | 4/-2 | 4 | 9/9 | DISCARD：挽救保留不足 |
| logistic-utility | 8/1 | 4/4 | 4/-3 | 6 | 9/9 | DISCARD：挽救保留不足 |
| mlp-utility | 11/3 | 2/4 | 9/-1 | 9 | 9/9 | DISCARD：挽救保留不足 |
| mlp-physical | 13/3 | 5/2 | 8/1 | 7 | 9/9 | KEEP |
| mlp-physical-utility | 9/0 | 1/2 | 8/-2 | 7 | 9/9 | DISCARD：挽救保留不足 |

多个discard版本虽然通过原九Gate，但未达到本轮预先冻结的挽救保留要求，未因此改标准晋级。没有以“减少commit次数”代替挽救效果。

## 3. 两个Development Block

Dev1：87000–87031；dev2：99000–99031。两个都是旧development块，不称fresh confirmation。dev2只有mlp-physical获得资格，新跑768 candidate episodes，未读取该块来重新拟合权重。

| 块 | 方法 | 总S/C/T，768例 | Square成功，384例 | Square低timeout | 挽救square/circle | 损害square/circle | 总平均原折扣return |
|---|---|---:|---:|---:|---:|---:|---:|
| dev1 | parent | 640/24/104 | 275 | 81 | 0/0 | 0/0 | 0.402961 |
| dev1 | v01 | 648/26/94 | 288 | 74 | 24/4 | 11/9 | 0.400351 |
| dev1 | candidate | 649/22/97 | 283 | 74 | 13/3 | 5/2 | 0.409353 |
| dev2 | parent | 677/23/68 | 312 | 54 | 0/0 | 0/0 | 0.434260 |
| dev2 | v01 | 671/22/75 | 311 | 55 | 11/4 | 12/9 | 0.418251 |
| dev2 | candidate | 682/22/64 | 316 | 50 | 7/3 | 3/2 | 0.435038 |

挽救交集单列：
- dev1：V0.1 square挽救24个，Candidate13个，其中12个确为保留、1个新增；circle保留3/4，未新增。总共保留15/28。
- dev2：V0.1 square挽救11个，Candidate7个，其中6个保留、1个新增；circle保留2/4、另新增1个。总共保留8/15。
- 因此没有把新挽救当作旧挽救保留，也没有声称“保留全部挽救”。

mlp-physical两个dev开发门槛都通过。dev1原九Gate为9/9；dev2为8/9，缺“两个人数配置均净改善”。dev2收益仅落在5人square，10/20人square与Parent终局相同，这在final前已明确记录，不包装为稳定密度泛化。

以下均为square“低timeout净减少/成功净增加”，负数表示恶化：

| Checkpoint seed | Dev1 | Dev2 | Final |
|---|---:|---:|---:|
| 419 | 2/1 | 3/3 | 0/-1 |
| 443 | 3/4 | 1/0 | 0/1 |
| 467 | 5/5 | 0/0 | -1/-1 |
| 491 | -3/-2 | 0/1 | 0/-2 |

| 人数 | Dev1 | Dev2 | Final |
|---|---:|---:|---:|
| 5 | 1/1 | 4/4 | 4/4 |
| 10 | 3/3 | 0/0 | -4/-4 |
| 20 | 3/4 | 0/0 | -1/-3 |

## 4. 唯一一次Fresh Final

代码/权重在读取final结果前锁定；140000–140031的unused检查无匹配。Parent/V0.1/Candidate × 四checkpoint × 5/10/20 × circle/square × 32case，共2304 episodes全部完成，没有选择性终止或重跑。

| 方法 | 总S/C/T，768例 | 总SR | Square S/C/T，384例 | Square低timeout | Circle S/C/T，384例 | 总平均原折扣return |
|---|---:|---:|---:|---:|---:|---:|
| Parent | 674/17/77 | 87.76% | 315/8/61 | 58 | 359/9/16 | 0.434014 |
| V0.1 | 674/18/76 | 87.76% | 316/10/58 | 55 | 358/8/18 | 0.424153 |
| Candidate | 673/18/77 | 87.63% | 312/10/62 | 59 | 361/8/15 | 0.431575 |

相对V0.1：
- Parent成功被破坏：square16→10，circle8→3，总24→13。确有减损害效果。
- square挽救17→7，其中真正保留6个、新增1个、丢失11个。
- circle挽救7→5，其中真正保留4个、新增1个、丢失3个。
- 总共保留V0.1原24个挽救中的10个，另新增2个；不是“全部保留”。
- 最终减少损害的同时丢失更多有效干预，无法满足主目标。

相对Parent：square成功315→312、低进度timeout58→59、碰撞8→10；circle成功359→361、碰撞9→8。overall原reward也下降。circle正效果不能抵销或改写预设square主目标。

| 原九Gate | Final实测 | 裁决 |
|---|---|---|
| Square低timeout严格净减少 | -1，58→59 | FAIL |
| ≥4不同合格square挽救case | 7个 | PASS |
| ≥2人数配置成功与低timeout同时改善 | 仅5人；10/20人恶化 | FAIL |
| ≥3/4 seed两项净方向非负 | 仅443满足 | FAIL |
| Square新增collision≤1pp | +2/384，+0.52pp | PASS |
| Square Parent成功损害≤2pp | 10/384，2.60pp，超过最多7个 | FAIL |
| Circle Parent成功损害≤2pp | 3/384，0.78pp | PASS |
| Circle SR净损失≤1pp | 成功净增2 | PASS |
| Circle新增collision≤1pp | -1/384 | PASS |

Final仅5/9通过，状态文件为FINAL_CONFIRMATION_FAILED。未因两开发块正反馈改成METHOD_CANDIDATE_FOUND。

每格128 episodes。到达时间仅对各臂成功episode求均值，成功集合不同，不作配对耗时因果证明；平均最小间距覆盖该格全部episode，不能替代碰撞计数。

| 人数 | Geometry | Parent S/C/T | Candidate S/C/T | 成功时间Parent/Candidate，秒 | 平均最小间距Parent/Candidate，m |
|---|---|---:|---:|---:|---:|
| 5 | circle | 121/0/7 | 121/0/7 | 15.05/14.98 | 0.5898/0.6003 |
| 5 | square | 110/0/18 | 114/0/14 | 15.15/15.38 | 0.5686/0.5605 |
| 10 | circle | 124/0/4 | 124/0/4 | 16.89/16.85 | 0.3557/0.3544 |
| 10 | square | 108/0/20 | 104/0/24 | 16.24/16.31 | 0.2749/0.2720 |
| 20 | circle | 114/9/5 | 116/8/4 | 20.15/20.15 | 0.1706/0.1711 |
| 20 | square | 97/8/23 | 94/10/24 | 17.95/18.00 | 0.1539/0.1500 |

## 5. 成本、正确性与保存

- 新标签采集435.3秒；11个gate fit合计2.62秒。
- 八个dev1候选闭环合计1083.1秒，4 workers；dev2 92.4秒、final 272.5秒，8 workers。均本地RTX3060，未下载服务器资产。
- 仅gate读出：最佳方法8.76µs/次，含方法调用，不含共同feature构造。权重193个float32，约772字节，另有标准化数组/容器。
- Final全策略平均推理耗时：Parent9.124ms、V0.1 9.187ms、Candidate9.562ms。包含GPU共享调度与不同轨迹，不当作独立硬件延迟的严格因果差。
- Final启动次数：V0.1 794，Candidate303；实际commit时间1364.75→460.50秒。少启动不是成功指标。
- Dev缓存latency在不同日期/worker负载下测得，不能直接与本次8-worker耗时比较来宣称效率收益。
- 单元测试228通过、3跳过；proposal-only重点测试4/4。三个原缓存完整轨迹上全接受=V0.1、全拒绝=Parent逐步一致。
- 所有8个dev1块、dev2与完整final均完成trace哈希、实际平滑动作、0.25秒时钟、原reward/discount、proposal-only initiation、安全release及gate输入输出检查；final共186037个控制步。43文件冻结绑定及四checkpoint哈希最终仍一致。
- 数据按case分组；不把同case的四权重轨迹当成四个独立场景。训练标签与部署feature分离；80动作候选不能写入gate或Parent真实记忆。

结果根目录：
/home/abc/workspace/shixu/outputs/autonav-overnight/

其中protocol、dataset、results.tsv、各版本源码/权重/experiment与offline记录、两个dev的summary/validation、final-lock/final-result和final summary可核查。原始counterfactual快照与所有episode trace本地保留；Git只归档代码、权重与紧凑证据，不声称原始大trace全部上传。

## 6. 失败原因与下一步裁决

已证实：
1. gate能在本轮offline与两个development块减损害、保留部分真实挽救。
2. 这个冻结候选不能在唯一fresh块兑现主目标；10/20人square和3个checkpoint出现负成功方向。
3. offline优势最高不代表闭环最好；复杂度最高也没有保证。未使用never-commit掩盖失败。
4. 更少干预虽然降低V0.1损害，但final同时丢失14个原V0.1挽救，留下10个square成功损害，整体未赢Parent。

有证据支持的推断：
- 本轮开发优势不足以证明gate稳定泛化；dev2已出现收益仅限5人square的预警。
- 受限五人circle标签、固定feature和监督目标下，此候选的accept/reject取舍不够可靠。不能用开发集正反馈替代确认。

尚未证明：
- 数据覆盖、跨密度特征外推、单次counterfactual训练与重复在线干预分布差异，分别贡献多少。未做新的根因实验，因此不指定其中之一为唯一原因。
- 所有合法gate都无法学习；或commitment机制本身无效。本轮不能推出这两种普遍结论。
- 方法新颖性与SCI充分性。本轮没有新增prior audit，不宣称首次自适应commit或新options理论。

唯一裁决：当前mlp-physical停止，不部署替代Parent；不按final调整feature、loss、阈值、duration或release，不再尝试第二个final候选。冻结负结果，等待用户决定是否制定全新的研究协议。动作持续性已有的局部挽救事实保留，但今晚没有交付通过fresh Gate的方法。

<!-- END PRESERVED SOURCE -->
