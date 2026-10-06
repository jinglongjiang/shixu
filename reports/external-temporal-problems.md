# External Temporal Problem Checks

PaS and interaction-response checks did not establish a repeatable method entry under their frozen protocols. Better occupancy estimates are not equivalent to safer navigation.

## Archive Policy

This is a classification-only consolidation. Original stage text, numbers, negative results and withdrawn claims are preserved byte-for-byte below. Later closure reports supersede earlier proposed next steps. The historical README is retained in thematic sections. Snapshot variants are labeled by their original paths. Frozen protocols and autoresearch_nav/program.md are not changed.

## Stage Index

- [PAS_TEMPORAL_PROBLEM_DISCOVERY.md](#stage-1)
- [pas-history-confirmation.md](#stage-2)
- [pas-failure-confirmation.md](#stage-3)
- [interaction-response-problem-audit.md](#stage-4)


---

<a id="stage-1"></a>

## Source: PAS_TEMPORAL_PROBLEM_DISCOVERY.md

Original full-source SHA-256: dee6909b7098d34cd2d300106ae6d378d62bbbaa6ddd9882ebd484abaeec018a

<!-- BEGIN PRESERVED SOURCE -->
# PaS Temporal Problem Discovery

日期：2026-10-05。范围：旧母体动作合同收尾，以及一个PaS入口的自然遮挡、合法历史和真实动作后果诊断。没有训练新模型，没有修改KDA，没有继续80009单点归因。

后续更新：独立32-case确认已完成，未重复终局挽救；10个合格root仅含1个原失败，正式状态为失败覆盖不足。当前不进入算法阶段、不自动扩样本。下面保留发现阶段原记录；最新确认及资源决策见pas-history-confirmation.md。

最新更新：随后完成方法盲的失败富集确认，8失败+8保护root五臂全部结束，长CV独有挽救0；停止当前投影/长CV入口，不以覆盖不足继续投入。长hold两例挽救保留，但有毫米级裕度和新增保护碰撞，未通过安全要求。最终结果与边界见pas-failure-confirmation.md；以下为发现阶段历史记录。

## 1. 本轮结论

**找到了一例合法较早历史改变动作、挽救真实终局的局部证据；实现它的是简单CV占据跟踪，不是KDA。**

20人square / case92001，在首次满足连续不可见至少2秒的预定root（tick42，10.50s）：原PaS超时；0.75秒记忆CV仍超时；4秒记忆CV仅改变首个0.25秒动作，随后恢复同一原策略，13.00秒后到达目标，最小表面间距0.083977m。

这足以保留“窗口外占据历史的动作价值”入口，**不足以宣布一般时序决策缺陷、新导航方法或复杂记忆增量已经成立**。当前信息不充分尚未通过匹配状态或反事实排除；原PaS本来就有持续GRU。没有训练过独立短窗/Current对照，也没有比较新GRU与KDA。

**当前决策：暂停KDA优先路线，保留PaS的简单历史占据验证。** 下一步只做独立配对确认，不启动新时序结构或大训练。

## 2. 旧母体工程收尾

独立分支：fix/executed-candidate-contract。已推送提交：9c7c62a。

shixu/policy.py先将每个grid index映射为实际将执行的平滑命令，再用同一命令计算successor、reward、clearance、filter、risk和value；选择后直接执行，不再二次平滑。训练阶段、首控制步和关闭平滑时保持原grid命令。动作编号、候选数量、奖励、权重及历史写入频率不变。

| 验证 | 结果 |
|---|---|
| 工程分支完整测试 | 168 passed，3 skipped；含5项新增合同测试 |
| 80009既有缓存回归，80动作 | successor/reward/clearance与既有executed-CV查询一致，全部分数最大误差0.0 |
| 选择及执行 | 仍选index17；执行命令(0.114768659, 0.449605519)，无重复平滑 |
| 历史及训练合同 | 评分不写真实历史；每个控制步写一次；训练命令不变 |
| 归档 | 109个远近文件SHA256一致；本轮自建4090 RAM目录已删除 |

这不是新闭环结果。缓存中的失败终局来自**原策略续跑**，不能称为整个修复策略的新终局验证。修复只解决已证实的工程合同问题，不解释KDA，也不补齐缺失RL replay。80009归因到此停止；旧审计和负结果原样保留。

## 3. PaS来源与冻结合同

使用/home/abc/temp/PaS_CrowdNav，提交eeaef4e的**官方代码派生、本地适配母体**。权重是本地seed1207、5人circle Standard训练15M PPO steps的last.pt，不是原论文权重，也不是完整论文复现。

原生链条：4帧合法sensor grid → Sensor-VAE latent → 持续FeatureRNN/GRU → 连续动作 → 原加速度/速度裁剪 → env.step。GRU拥有窗口之前的状态，不能把4帧输入窗口说成整套策略只有1秒记忆。

保持本地母体原奖励、动力学、Standard可见性、grid分辨率、0.25秒时钟及动作支持；不额外加入遮挡、噪声或延迟。PaS动作不是shixu的80动作grid；本轮有限headroom使用PaS原连续动作支持。

当前适配sensor在generate_ob中把不可见行人的footprint写为0.5。因此unknown图形本身可能携带几何结构。**不把“当前不可见”直接等同于“当前输入完全没有位置线索”，不把本地sensor协议当成原论文精确协议。**

每个分支只替换root第一步命令，保留原full4 root的GRU更新及后续随机流；之后使用原full4策略。原控制器的加速度、速度裁剪完整保留，记录真正执行速度，不用raw action差异冒充执行差异。累计原奖励，gamma=0.99，直至原成功/碰撞/超时终止；不是Q*。

## 4. 自然事件覆盖与root选择

先重放原8条91000/91001完整指令轨迹，指令、奖励、终止一致。以前其他正式评估资产只有episode摘要，不能恢复完整观察；本轮新采集不冒充旧RL replay。

补采规则在结果之前冻结：10/20人 × circle/square × case92000至92003，共16条。总计24条、1529控制帧；原策略22成功、1碰撞、1超时。它们是覆盖检查，不是随机总体SR估计。

| 自然事件，actor中心距机器人≤3m | 控制帧数 | 说明 |
|---|---:|---|
| 当前不可见 | 559 | 每帧至少一人；不是559个独立样本 |
| 当前不可见且从未见过 | 37 | 历史没有见过的目标不能靠记忆恢复 |
| 不在最近4帧可见记录中 | 239 | 含从未见过 |
| 曾见过，现不在最近4帧中 | 204 | 合法较早历史可能有内容 |
| 曾见过，距末次可见≥2秒 | 72 | 8个episode出现，其中原策略7成功、1超时；与上面计数重叠 |

第一阶段按“首次t≥2s、近处不可见且窗口外”选root：16个episode合格，仅一个失败episode合格。按固定队列取该失败和前3个成功root。第一次干预后发现这些root尚未连续遮挡满2秒，**不能据此关闭长期遮挡机制**。

于是单独冻结覆盖补查：只在原24条轨迹中取“首次t≥2s、曾见过且连续不可见≥2秒”的root，仍按队列取失败及前3个成功。规则在补查干预前冻结，未新增场景、未按哪个时刻有收益替换root。补查与第一阶段共用episode，**不是独立确认集**。成功对照也不是人数/几何严格匹配对照。

## 5. 合法历史与共同消费者

匿名GridTracks只读取sensor中已测occupied cell的连通区域中心；用机器人自身坐标图映射到世界坐标。Hungarian预测距离关联，gate固定1m；速度来自两次合法中心测量。未知cell不是测量，失踪不会重写位置/速度/测量时间。不读取真actor ID、目标、真速度或未来位置。

比较4秒CV、0.75秒CV及4秒原位置保持。不调TTL。根查询只读跟踪状态，输出假设占据，仅修改当前unknown cell，不提交为真实传感器测量。真ID只用于离线事件标注，真占据只用于oracle分支及误差标签。

共有8臂：原full4、short2、current-grid；另5臂通过同一个冻结Label-VAE读取原decoded占据、短CV、长CV、长hold或真占据。short2/current-grid保留既有GRU，不是新训练的无记忆策略。

**消费者限制：** 5臂的Label-VAE投影不同于原Sensor-VAE读入。projected-parent单独列出以暴露该变化；概率占据图与训练输入也可能不同。short-CV与long-CV在同一投影消费者中比较，仍只能支持该消费者下的局部历史作用，不证明其他消费方式或原GRU内部缺陷。匿名关联有误配风险，未宣称完美恢复身份。

## 6. 两阶段真实后果

### 6.1 第一阶段：窗口外但尚未遮挡满2秒

失败root为20人square / 92001 / tick10，2.50s。原full4超时，Q=3.805428，剩余47.50s；其余7臂全部碰撞，包括CV与真占据。部分碰撞Q反而更高，**不计为改善**。

有限34动作（原命令、stop、16方向×2速度）有2成功、25碰撞、7超时，证明局部动作headroom；不证明历史能够利用它。另3个原成功root的8臂全部成功，但回报/间距并非全部改善。这批负结果不因后续正例撤回。

### 6.2 覆盖补查：首次连续不可见至少2秒

同一失败episode，root=tick42，10.50s。近处不可见actor8曾被看到、已连续不可见至少2秒。下面全部使用同一原策略续跑，时间/间距只统计root后。

| root干预 | 真实终局 | 原折扣Q | 剩余秒 | 最小表面间距m | 隐藏occupied cell召回 |
|---|---|---:|---:|---:|---:|
| 原full4 | Timeout | -2.419595 | 39.50 | 0.002772 | 不适用 |
| short2，保留GRU | Collision | -1.521729 | 8.00 | -0.010257 | 不适用 |
| current-grid，保留GRU | Collision | -2.758602 | 11.25 | -0.009219 | 不适用 |
| projected-parent | Collision | -1.483084 | 9.00 | -0.015467 | 0.00% |
| 0.75秒CV | Timeout | -2.087008 | 39.50 | 0.060067 | 30.75% |
| **4秒CV** | **ReachGoal** | **7.810495** | **13.00** | **0.083977** | **58.07%** |
| 4秒hold | Collision | -1.032606 | 7.50 | -0.101857 | 41.93% |
| 真占据，离线诊断 | Collision | -2.457352 | 11.00 | -0.003370 | 100.00% |

长CV相对原full4的ΔQ=+10.230090，相对短CV为+9.897503；hidden-occupied MSE由短CV的0.692160降至0.419250。长CV不是只改raw动作：实际首步速度从原(0.566140,0.824309)变为(0.500279,0.865864)；短CV执行(0.529752,0.848153)。这次是真实终局改善，不是仅预测更准或动作有变化。

该root34个有限候选有6成功、27碰撞、1超时。长CV成功不等于选择全局最优动作，也不证明收益具体来自actor8而非多个历史占据变化的组合。

**反例保留：** 真占据更准，却碰撞；长hold保留历史，也碰撞。因此不能宣称信息精度单调决定动作质量。真占据的负结果只能否定这次冻结投影用法，不能否定整个历史问题。投影父臂本身由原超时变为碰撞，说明消费者兼容性不能忽略。

### 6.3 原成功case的保护检查

补查的3个成功root全部8臂仍成功；但长CV不是一致占优。

| 原成功case / root秒 | 长CV减原Q | 原/长CV最小间距m | 原/长CV剩余秒 |
|---|---:|---|---|
| 10人circle91001 / 5.50 | +0.189501 | 0.100724 / 0.101988 | 9.50 / 9.75 |
| 10人circle92001 / 5.25 | -0.003353 | 0.351236 / 0.350795 | 7.50 / 7.50 |
| 10人circle92003 / 7.50 | -0.858760 | 0.127490 / 0.044750 | 8.75 / 8.25 |

最后一例回报和间距变差，即使仍成功，也不能说没有安全/效率代价。失败只一例、对照只三例，且均为选定root，不能计算或宣传总体SR提升、跨seed稳定收益或5→10/20人方法泛化。

## 7. 事实、推断与未证明项

| 项目 | 证据等级 | 裁决 |
|---|---|---|
| 自然窗口外/长期不可见事件确实发生 | 已证实 | 24条轨迹及固定事件规则可重放；不是人为新加遮挡 |
| 较早合法观测能恢复部分当前未知占据 | 已证实，所测root | 简单CV提高hidden recall、降低MSE；没有使用真ID/真运动输入 |
| 历史经共同消费者能改善真实动作后果 | 已证实，单个指定状态 | 4秒CV成功，原策略及0.75秒CV超时；其余首步合同相同 |
| 该状态存在动作改善空间 | 已证实，指定续跑 | 两阶段有限动作集分别有2/6成功；不是Q* |
| 长期占据保持是值得检验的能力 | 有证据支持的推断 | 正例和短CV/hold差异支持后续独立确认，不等于一般性机制证明 |
| 当前合法输入本身不充分 | 尚未证明 | 未做当前特征严格匹配/反事实；unknown图形、已有GRU等线索不能忽略 |
| 原GRU忘记actor8导致超时 | 尚未证明 | 未定位GRU编码或孤立该actor的因果作用；不得由一个失败推成GRU无效 |
| 简单CV已解决整个问题 | 尚未证明 | 一例挽救、另root失败、成功对照有回报/间距损失 |
| KDA或复杂memory具有独立增量 | 无证据 | 本轮未运行KDA、未训练新GRU；不能把简单历史收益记到KDA |
| 已具备正式方法立项/论文证据 | 尚未达到 | 消费者兼容、自然复现频率、独立确认及强训练对照仍缺失 |

## 8. 唯一下一步与停止条件

**下一步只做预冻结独立PaS配对确认：验证简单占据历史的收益能否重复且不破坏原成功。** 保持相同母体、sensor、权重、共同投影消费者和首次长期遮挡root规则；比较原full4、projected-parent、短CV、长CV及hold，不能在确认case上改TTL/阈值/权重。必须同时报告成功、碰撞、超时、原Q、间距及额外计算。先实测一块耗时，再冻结小样本数量。

这一轮先不执行追加确认或训练。无需先要求GRU失败才承认局部历史价值，但当前简单CV已经实现正例，不能据此跳到复杂模型。

若收益在独立case不重复，或明显增加碰撞/破坏原成功，则停止**当前投影+CV实现**；不自动否定所有长期历史问题。若简单CV稳定吸收收益，则优先用简单办法，复杂时序无新增立项依据。只有找到可部署历史能力、可兼容消费者以及简单办法留下的可检验增量，才讨论GRU/KDA等架构公平原型；不以模型名字重启旧V1–V8。

## 9. 归档与运行

PaS研究代码独立于工程修复分支；没有修改PaS源码或冻结权重。新增测试覆盖未知cell不充当测量、合法速度来源、缺失不写伪观测、TTL及只读查询。研究分支完整测试：168 passed，3 skipped。跳过项为可选旧legacy parity依赖，不冒充已通过。

正式测试命令为下列显式源码范围。第一次不限定范围时误收集outputs内的旧备份测试；第二次未指定vendor路径时误加载全局安装的旧camrl/crowd_sim，3项环境测试失败。纠正测试入口后全部通过，没有修改源码来绕过失败。PaS诊断则独立运行，不使用shixu/vendor的PYTHONPATH，避免混用同名包。

```bash
PYTHONPATH=/home/abc/workspace/shixu/vendor:/home/abc/workspace/shixu python -m pytest -q tests
```

运行环境：本地Python3.8.10、Torch2.1.0+cu121，PaS诊断在CPU。补查4个root全部8臂和失败root34候选已完成，无未结束任务。耗时记录是含真实续跑的诊断耗时，不是在线推理延迟；尚未测量可部署方案的额外成本。未作新的文献新颖性claim。

| 冻结资产 | SHA256 |
|---|---|
| PaS checkpoint | 07b9703e18204089b4318fd2c157ada4702013b4c88b75033c55a8773adbf169 |
| PaS config | 7403a4a75d121af7ae01ff5ce286dc57efedcd1e3c8d0ff54f259192ef562a83 |
| pas_problem_discovery.py | 219fe4d1da5bfd04a9c5db57f41357067bd35bc8cc600e9b3d494bc22635a76c |
| pas_long_occlusion.py | 738b39f8dee1bf6455ca5db62b5c5676d49071c84ec34bcbf40cae10827ccdc8 |

protocol.json保存预设规则、checkpoint/config及全部母体Python文件哈希；补查protocol保存原24份case哈希。最后核对原8条资产、母体源码、权重和两个脚本均未变化。archive_verified.json逐项保存63份归档文件哈希，不覆盖原结果。

本地证据目录：

/home/abc/workspace/shixu/outputs/pas_problem_discovery/

/home/abc/workspace/shixu/outputs/pas_problem_discovery/long_occlusion/

/home/abc/workspace/shixu/outputs/execution_contract_fix/

复现入口（PaS本地依赖及冻结资产需存在，不将大权重/轨迹上传Git）：

```bash
python experiments/pas_problem_discovery.py prepare
python experiments/pas_problem_discovery.py collect
python experiments/pas_problem_discovery.py diagnose
python experiments/pas_long_occlusion.py prepare
python experiments/pas_long_occlusion.py evaluate
```

prepare拒绝覆盖冻结协议；collect/diagnose/evaluate跳过已完成结果。复算应使用保留原证据的独立工作副本；不能删除旧负结果来使脚本重跑。

<!-- END PRESERVED SOURCE -->


---

<a id="stage-2"></a>

## Source: pas-history-confirmation.md

Original full-source SHA-256: c4d7cf95a718bf0f9a3039f448a6ebb723951b729ecb5017011965d198de1224

<!-- BEGIN PRESERVED SOURCE -->
# PaS Long-History Problem Confirmation

日期：2026-10-05。阶段：独立case问题确认，不是算法开发。本轮没有训练、调参、更换模型、增加人为遮挡或重新拆解92001。

后续更新：方法盲的自然失败确认已完成，88条parent轨迹中冻结8失败+8保护root；长CV独有挽救0，失败覆盖不足的问题已补齐。当前停止这套冻结长CV入口，不启动模型研发。下面保留本批32-case原记录；最新完整结果见pas-failure-confirmation.md。

## 1. 结论

**这批独立case没有复现“短CV失败、长CV成功”；不具备进入算法阶段的依据。**

32个新case中，10个出现预定长期遮挡root，其中原PaS为9成功、1碰撞。五臂在这10个root的终局完全一致：9成功、1碰撞；挽救0，原成功转失败0。长CV的占据估计确实更好，但没有转成可重复终局收益，原Q也没有整体改善。

正式诊断状态：**INCONCLUSIVE_FAILURE_COVERAGE**。只出现1个合格失败，未达到冻结协议需要的至少2个重复挽救机会，不能把“没复现”升级成“时序问题不存在”。旧92001正例保留，但仍是未重复的局部观察，不再追加解释来包装它。

**资源决策：本批结束，暂停当前投影+长CV入口的算法化投入；不启动KDA/GRU训练，不自动扩大case，不调TTL救结果。** 这不是永久否定长期占据记忆，而是现有证据不足以继续开发这版用途。

## 2. 冻结合同与独立性

- 同一本地适配PaS母体、seed1207 checkpoint及Standard sensor；不是原论文权重或完整baseline复现。
- 10/20人 × circle/square × case93000至93007，共32个此前未用于问题发现的配置/case组合；既有正式评估摘要中也没有这些case ID。
- 8个scene seed编号跨4个任务配置复用，不能把32个组合当成32个独立同分布样本；没有更换training seed或checkpoint。
- 全部32个case在测试前登记，不根据失败数量或方法结果扩样本。
- root仍为首次t≥2s、曾见过且连续不可见≥2s、当前中心距机器人≤3m的状态；没有root就保留原终局，不换case、不另挑时刻。
- 0.75秒CV、4秒CV、4秒hold、1m关联gate及共同Label-VAE消费者完全不变。
- root只改变首个0.25秒动作；之后共享原full4策略、原root GRU更新及对应随机流。原奖励、gamma=0.99、加速度/速度裁剪、终止判据不变。

五臂为原full4、projected-parent、short-CV、long-CV、long-hold。投影父臂用于显式暴露换读入接口的影响。未添加真值动作臂、动作穷举或新消费者。

冻结的确认标准：至少2个不同新case中，原PaS、projected-parent及short-CV都失败，而long-CV成功；同时不得新增加原成功case的碰撞，不得增加相对原PaS/短CV的总体碰撞，并须有净成功增益。安全裕度和原Q下降仍需报告。该标准是小样本研发确认，不是论文级统计检验。

## 3. 覆盖和真实终局

32条原轨迹共1700控制帧，原PaS27成功、5碰撞、0超时。10条出现合格root，另22条未干预。**实际配对后果比较为10个root×5臂，不是32个root×5臂。**

| 范围 | 原成功 | 原碰撞 | 原超时 | 说明 |
|---|---:|---:|---:|---|
| 全部新case | 27 | 5 | 0 | 固定有限集合的描述，不作总体SR推断 |
| 有合格长期遮挡root | 9 | 1 | 0 | 这10个root全部完成五臂 |
| 没有合格root | 18 | 4 | 0 | 未发生规定干预，不能算失败修复试验 |

| 五臂，10个配对root | 成功/碰撞/超时 | 平均原Q | 平均剩余秒 | 平均最小间距m |
|---|---|---:|---:|---:|
| 原full4 | 9 / 1 / 0 | 14.618416 | 6.750 | 0.410720 |
| projected-parent | 9 / 1 / 0 | 14.586060 | 6.775 | 0.394492 |
| 0.75秒CV | 9 / 1 / 0 | 14.575936 | 6.750 | 0.402320 |
| 4秒CV | 9 / 1 / 0 | 14.539873 | 6.725 | 0.402113 |
| 4秒hold | 9 / 1 / 0 | 14.582034 | 6.750 | 0.401378 |

上述均值包含1次碰撞，时间/间距只统计root之后，不能当作完整episode平均到达时间或安全证明。没有把更高Q的碰撞判为收益。

唯一合格失败为20人square / 93000 / tick29（7.25s）。原、投影、短CV、长CV、hold均在后续5.25秒内碰撞。原Q=1.135012，短CV=1.133303，长CV=1.131678；本轮没有额外穷举该状态的action headroom，因此也不声称这里根本不存在好动作。

## 4. 长历史究竟增加了什么

新增内容来自匿名tracker的较早合法occupied-cell观测及其CV外推，不是隐藏目标、真ID、真速度或未来位置。相对短CV，长CV仅在当前unknown cells加入占据假设。

| 10个root的平均占据误差 | hidden-occupied MSE | hidden召回率 | unknown Brier |
|---|---:|---:|---:|
| projected-parent | 0.990079 | 0.59% | 0.060739 |
| 0.75秒CV | 0.765066 | 23.09% | 0.049937 |
| 4秒CV | 0.559602 | 43.84% | 0.045274 |
| 4秒hold | 0.776745 | 21.92% | 0.068368 |

长CV在8/10个root降低hidden-occupied MSE，另2个不变。长短差异共增加713个占据cell，其中441个按离线真值确实被占据，272个实际为空。真值只用于事后计分，不参与可部署输入。额外cell不是独立样本，也不能把false occupancy唯一归因为陈旧记忆：关联、CV外推及grid误差尚未分离。

因此，**“长历史恢复了更多占据”得到支持；“恢复的占据重复改善了终局”没有得到支持。** 两者不能互相代替。

## 5. 安全、进度与原Q

原成功的9个root全部仍成功，没有新增碰撞或超时；但不能因此说没有代价。

长CV相对短CV的Q只在2/10个root上升、7个下降、1个相同，平均差-0.036063。相对原full4的平均差-0.078543。长CV与短CV在原成功root中，最小间距3个下降、3个上升、3个相同，没有一致安全提升。

例：20人square / 93003，短CV的最小间距0.111074m，长CV0.095075m，Q差-0.361291，虽然都成功，仍是反向证据。10人circle / 93006相对原PaS的间距也下降，但projected-parent已经从0.192949m降到0.108565m，长CV为0.108778m；不能把这项完整下降都归因于长历史。

没有把剩余时间平均减少0.025秒包装成效率收益。终局一致、Q并不整体改善、裕度有升有降，这批不满足预定正向确认。

## 6. 已证实与尚未证明

| 问题 | 本轮证据 |
|---|---|
| 旧正例能否在新case重复 | 未重复；合格失败仅1例，失败覆盖不足 |
| 是否增加碰撞/破坏成功 | 所测9个原成功root没有终局破坏；不等于风险为零 |
| 合法较早历史是否增加占据信息 | 是，8/10个root的hidden MSE下降，但有272个新增false occupied cell |
| 更多占据是否足以改善动作后果 | 本批未建立；占据改善与终局/Q增益分离 |
| 原PaS的GRU是否存在通用长期记忆缺陷 | 尚未证明；持续GRU本来就存在，消费者兼容性仍有边界 |
| 是否该上KDA或另一种新模型 | 否，没有新的方法立项依据 |

不得把本批阴性结果扩成“长期历史一般无价值”，也不得因为存在旧正例继续随机加case或调参。当前投影消费者的结果只适用于冻结的条件和续跑策略。

## 7. 复现与交付

新增入口：experiments/pas_history_confirmation.py；新增3项测试，核对固定case队列、五臂一致及额外占据统计不修改合法输入。完整tests为171 passed、3 skipped，3项仍是旧可选legacy依赖。

本地CPU四进程、每进程一个Torch线程；32个完整配对块墙钟35.46秒，不含实现、核查和测试。这不是在线推理延迟。长CV单次占据读入查询的中位时间5.31ms，不含tracker更新/占据构建，并受并行CPU负载影响，不能用它宣布部署效率优势。

冻结源码、母体、checkpoint和旧资产最后再次核验；结果保存在新目录，不覆盖发现阶段的正例/负例。summary.json保存病例统计、逐root后果及66份输入/轨迹/结果哈希；所有32份case JSON完整可读取。

| 资产 | SHA256 |
|---|---|
| PaS checkpoint，未变 | 07b9703e18204089b4318fd2c157ada4702013b4c88b75033c55a8773adbf169 |
| 本轮protocol | 58c6e6e3587705c2a80efff41ad52e265a8092bdeb68d4a9541285958418a7b5 |
| 本轮summary | 6fb28073474508bc87be7c9584c4e61b8150439a86528e0c161d7ba4a28fe3da |

结果目录：

/home/abc/workspace/shixu/outputs/pas_history_confirmation/

```bash
python -m experiments.pas_history_confirmation prepare
python -m experiments.pas_history_confirmation evaluate --workers 4
```

PaS入口需独立运行，不把shixu/vendor放到PYTHONPATH。已有结果与冻结protocol不允许覆盖；脚本可跳过已完成case恢复未完成块。全部32块已经结束，没有未完成的有利/不利对照。

**下一步唯一决策：归档并停止这一批，不上算法，不自动开启下一轮搜索。** 若以后重新授权确认，必须事先定义能增加合格失败覆盖、但不按长CV收益选样本的独立协议；不能用本批case调规则后仍叫独立确认。

<!-- END PRESERVED SOURCE -->


---

<a id="stage-3"></a>

## Source: pas-failure-confirmation.md

Original full-source SHA-256: 4a4c3d8e8612a50b90913377dc5f020a420dfe2a407871cfa710a5e310121fd1

<!-- BEGIN PRESERVED SOURCE -->
# PaS Long-History Natural-Failure Confirmation

日期：2026-10-05。任务：按原PaS结果盲选配置与失败root，完成一次有停止条件的五臂确认。没有训练、调参、新消费者、真值动作臂或KDA实现。

## 1. 最终结论

**STOP_FROZEN_LONG_CV_ENTRY：停止当前投影消费者下的长CV占据入口，不进入算法开发。**

本轮已补齐此前缺少的8个自然长期遮挡失败机会，以及8个同配置成功保护root。长CV没有一例满足“原PaS、投影父臂、短CV都失败，而长CV成功”的预定条件，不再以失败覆盖不足作为继续理由。

但不能写成“五臂完全一样”或“历史没有任何动作价值”：

- 长CV在95085挽救原超时，但投影父臂和短CV也成功，且更快、Q更高。不是长历史独有收益。
- 95015原PaS成功，投影和短CV变成超时，长CV恢复成功。是对替换读入接口造成损失的保护，不是原PaS失败挽救。
- 长hold在95010、95078确实把原碰撞变成成功，但最小间距仅0.004001m、0.003218m，未达到事先冻结的0.05m安全裕度；还把95030原成功变成碰撞。
- 长CV在14/16个root降低隐藏占据误差，另2个不变；信息改善仍没有达到预定、安全可重复的失败挽救要求。

**资源决策：归档并停止当前PaS用途及其在shixu上的延伸，不追加TTL搜索、不换成hold救裁决、不启动GRU/KDA训练。** 这是一项对具体实现的停止决策，不是证明所有PaS、shixu或时序导航方法无效。

## 2. 方法盲的配置选择与停止

只读取既有56条发现/确认轨迹的parent配置、终局及长期遮挡计数，没有读取任何干预臂的动作、Q或收益。按“长期遮挡且parent失败”的比例、数量排序，再用parent失败率和遮挡覆盖率打破并列。

| 原生配置 | 既有episode | parent失败 | 长期遮挡 | 两者同时发生 |
|---|---:|---:|---:|---:|
| 5人circle | 2 | 0 | 0 | 0 |
| 5人square | 2 | 1 | 0 | 0 |
| 10人circle | 14 | 0 | 5 | 0 |
| 10人square | 14 | 2 | 0 | 0 |
| 20人circle | 12 | 0 | 6 | 0 |
| 20人square | 12 | 4 | 7 | 2 |

只有20人square符合联合条件，因此只选这一个配置，没有为了凑两个配置继续搜索。新队列预先固定为case95000至95159，最多160个episode；筛选阶段只运行parent。

实际收集88条parent轨迹、4209控制帧：41成功、43碰撞、4超时。第8个合格失败出现在case95085，选择前缀为95000至95085，共86条；四进程批次中已在途的95086、95087完成后保存，但排除在root选择之外。没有跑满160，也没有干预结果驱动的补样本。

冻结失败case：95006、95010、95026、95032、95041、95047、95078、95085。

冻结成功保护case：95004、95005、95013、95015、95020、95030、95033、95035。均为同一20人square配置、队列中最早的8个合格parent成功。

每个episode只取首次t>=2s、曾见过且已不可见至少2s、当前中心距机器人<=3m的root。没有因方法表现更换时刻。选择文件及88份parent记录的哈希在五臂评估前冻结。

**baseline失败加遮挡并不自动证明失败由遮挡引起，也不保证首动作可挽救。** 本轮补齐的是自然失败机会，不是8个已确认的action oracle headroom。

## 3. 冻结合同

使用官方代码派生的本地适配PaS，提交eeaef4e；checkpoint为本地seed1207、5人circle Standard训练15M PPO steps的last.pt。不是原论文权重，不是完整论文复现。

保持原checkpoint、奖励、gamma=0.99、动力学、0.25s控制步、可见性、grid、加速度/速度裁剪及终止条件。每个root只改变首个0.25s动作，之后共享原full4策略、原root GRU更新及对应随机流，真实模拟器续跑至终止。Q是这个续跑策略下的实际折扣奖励，不是Q*，不是进度替代奖励。

| 臂 | 输入/作用 |
|---|---|
| full4 | 原4帧Sensor-VAE及原策略GRU |
| projected-parent | 原解码占据经共同Label-VAE读入，不加tracker历史 |
| short-CV | 同一投影接口，加0.75s匿名CV占据记忆 |
| long-CV | 同一投影接口，加4s匿名CV占据记忆 |
| long-hold | 同一投影接口，加4s匿名位置保持记忆 |

tracker关联gate固定1m，只读取合法sensor occupied cells及机器人自身坐标图。真ID、隐藏位置只用于离线root资格与误差计分，不进入tracker和策略输入。母体GRU原本就持续保留状态；full4不能称为没有长历史的模型。

本地sensor对不可见行人的footprint标unknown，unknown形状可能已有几何线索；Label-VAE替换Sensor-VAE也可能产生输入分布变化。这些既有边界不因本轮补足失败样本而消失。

## 4. 16个root的真实终局

表中S=ReachGoal，C=Collision，T=Timeout。分组依据冻结的原PaS终局，不能把此选择集当作随机总体SR测试。

| 组别 | case | root时间s | full4 | 投影 | 短CV | 长CV | 长hold |
|---|---:|---:|---|---|---|---|---|
| 失败 | 95006 | 5.50 | C | C | C | C | C |
| 失败 | 95010 | 5.75 | C | C | C | C | S |
| 失败 | 95026 | 6.75 | C | C | C | C | C |
| 失败 | 95032 | 20.75 | T | T | T | T | T |
| 失败 | 95041 | 21.75 | T | T | C | T | C |
| 失败 | 95047 | 5.25 | C | C | C | C | C |
| 失败 | 95078 | 6.00 | C | C | C | C | S |
| 失败 | 95085 | 19.25 | T | S | S | S | S |
| 保护 | 95004 | 10.25 | S | S | S | S | S |
| 保护 | 95005 | 9.75 | S | S | S | S | S |
| 保护 | 95013 | 7.25 | S | S | S | S | S |
| 保护 | 95015 | 22.50 | S | T | T | S | S |
| 保护 | 95020 | 8.00 | S | S | S | S | S |
| 保护 | 95030 | 8.00 | S | C | S | S | C |
| 保护 | 95033 | 4.50 | S | S | S | S | S |
| 保护 | 95035 | 8.00 | S | S | S | S | S |

| 全16个root | 成功/碰撞/超时 | 平均原Q | 平均剩余秒 | 平均最小间距m |
|---|---|---:|---:|---:|
| full4 | 8 / 5 / 3 | 7.392825 | 12.109375 | 0.306324 |
| projected-parent | 7 / 6 / 3 | 6.255650 | 11.531250 | 0.297260 |
| short-CV | 8 / 6 / 2 | 6.888642 | 11.453125 | 0.281499 |
| long-CV | 9 / 5 / 2 | 7.487523 | 11.421875 | 0.291684 |
| long-hold | 10 / 5 / 1 | 7.718716 | 10.921875 | 0.279026 |

均值含碰撞/超时，不是成功episode平均到达时间，不能只凭均值宣布方法获益。全部80个配对续跑完成，无未完成的不利臂。

## 5. 关键收益和代价不能混算

| case/臂 | 终局 | 原Q | 剩余秒 | 最小间距m | 判读 |
|---|---|---:|---:|---:|---|
| 95085 full4 | 超时 | -0.602657 | 30.75 | 0.672190 | 原失败 |
| 95085 投影/短CV | 成功 | 5.787987 | 13.00 | 0.672190 | 不需要长CV也能挽救 |
| 95085 长CV | 成功 | 4.100894 | 20.00 | 0.563394 | 不构成长历史独有增量 |
| 95015 full4 | 成功 | 6.496892 | 9.50 | 0.180771 | 成功保护root |
| 95015 投影/短CV | 超时 | -1.041858 | 27.50 | 0.179309 | 新读入接口损失 |
| 95015 长CV | 成功 | 5.891864 | 11.00 | 0.180998 | 恢复成功，仍比原Q低/更慢 |
| 95010 长hold | 成功 | 12.850691 | 9.25 | 0.004001 | 碰撞被挽救，但裕度不足 |
| 95078 长hold | 成功 | 9.299417 | 17.75 | 0.003218 | 碰撞被挽救，但裕度不足 |
| 95030 full4 | 成功 | 12.565951 | 13.50 | 0.103146 | 成功保护root |
| 95030 长hold | 碰撞 | -1.346682 | 6.00 | -0.033303 | 保护失败，不能以净成功掩盖 |

95041中短CV把原超时变为碰撞，长CV仍超时，Q相对短CV提高3.056341。这是避免一种更坏终局，不是成功挽救。

长CV相对短CV的Q为7升、8降、1相同，平均差+0.598881；相对full4平均差+0.094698。Q平均为正不能替代未通过的挽救条件，也不能说“Q全部不改善”。在8个原成功保护root中，长CV平均Q=14.137580，低于原full4的14.313632；平均剩余时间8.5625s，也长于原full4的8.34375s。

## 6. 提前冻结的裁决

五臂运行前已冻结：至少2个不同原失败case，full4/投影/短CV均失败而长CV成功；计入挽救的最小间距>=0.05m；8个保护root不得新增碰撞/超时；总体碰撞不得超过full4或短CV；保护root相对full4或短CV不得损失>=0.05m间距。0.05m是本轮工程安全门槛，不是普适社会安全定义，也没有用于调方法参数。

| 条件 | 结果 |
|---|---|
| 8失败+8保护覆盖 | 满足 |
| 长CV独有挽救>=2 | **0，失败** |
| 计入挽救的裕度>=0.05m | 没有符合资格的挽救 |
| 长CV保护集新增collision/timeout=0 | 满足，8个都成功 |
| 长CV总体碰撞不增加 | 满足，5次；full4为5、短CV为6 |
| 长CV保护集间距损失<0.05m | 满足，最大损失0.027691m |
| 全部条件共同通过 | **否** |

**不存在失败覆盖不足的退出借口，也不允许把long-hold的两例结果临时替换long-CV的主判据。** long-hold本身有新增保护碰撞和毫米级挽救裕度，不能升级为合格替代入口。

## 7. 长历史恢复了什么

| 16个root的平均误差 | hidden-occupied MSE | hidden召回率 | unknown Brier |
|---|---:|---:|---:|
| projected-parent | 0.985437 | 1.22% | 0.056757 |
| short-CV | 0.794342 | 20.33% | 0.046463 |
| long-CV | 0.578652 | 41.92% | 0.042279 |
| long-hold | 0.761282 | 23.66% | 0.061001 |

长CV相对短CV在14/16个root降低隐藏占据MSE，2个相同。新增1216个占据cell：723个真占据、493个实际为空。真值仅用于离线计分，cell不是独立样本；空cell增加不能唯一归因为陈旧记忆，关联/CV/栅格误差未分离。

因此，“较早合法历史能恢复更多隐藏占据”有直接证据；“这套长CV读入在困难自然root上带来满足安全要求的独有导航挽救”未通过。不能用前者替代后者。

## 8. 正确性、工程纠正与复现

两次工程纠正完整保留原协议与前后源码：

1. NumPy scalar不能直接写JSON。仅补标量序列化，已完成原轨迹从PT恢复，不重新运行parent，不改配置/选择。
2. 原生Policy.act动作是float32，JSON重读默认float64。在95032、95041、95085、95015回放时，原动作查询超过原1e-7一致性阈值。恢复float32后，动作及执行速度精确一致。没有放宽阈值，没有修改冻结母体或旧helper。

最终16个root逐项核对：世界状态、vector/grid/label-grid、前后GRU hidden、前后Torch RNG精确一致；full4整条剩余command和实际executed velocity与筛选记录精确一致，Q与原保存reward递推一致。已完成10个配对块核对后保留，补齐其余6个；没有覆盖旧发现/确认结果。

新增7项测试：parent-only配置排序、首8失败与在途排除、非遮挡失败排除、保护配置配额、cap停止、NumPy序列化、动作dtype恢复。完整tests：**178 passed，3 skipped**；skip仍是旧可选legacy依赖。

本地CPU四进程、各1个Torch线程，没有4090下载/训练。恢复后的筛选调用墙钟56.39s，补齐评估调用墙钟32.76s，均不含此前中断、实现和核查，不能相加冒充全任务耗时。16个完整配对块各自记录耗时合计201.99s；并行进程耗时之和不是墙钟，也不是在线推理延迟。本轮没有建立部署效率收益。

| 资产 | SHA256 |
|---|---|
| checkpoint | 07b9703e18204089b4318fd2c157ada4702013b4c88b75033c55a8773adbf169 |
| protocol | 4b268e4a7924f682c4c051786c73251adec1b259e758a18822f09045082bf6eb |
| frozen selection | 95f5ee0ddecc11992272bce25680da61d9f45530ada18a7a73c31b0493c9d2a6 |
| completion | 9acc435426a919e32194c6cb14c0018b2fc23eec46083f7a6c69607ff9650132 |
| summary | 81638695cc59421633c341e2af1988cb5e3707756075dd823ca239df5be10797 |
| corrected diagnostic source | d99b775054d2c6a1a8d6e8b6fd1b42ee835ec46ad7071053a3b887d7303b4e4d |

结果目录：

/home/abc/workspace/shixu/outputs/pas_failure_confirmation/

入口：experiments/pas_failure_confirmation.py。独立执行，不把shixu/vendor加入PaS运行的PYTHONPATH。已有协议/选择/结果禁止覆盖；已完成块可以校验并跳过。protocol、selection、summary、88份原轨迹、16份root与五臂结果均本地保存；原56份资产及权重哈希最后再次核验。

## 9. 已证实、未证明与唯一下一步

| 等级 | 内容 |
|---|---|
| 已证实 | 有8个合格自然失败；五臂全完成；长CV独有挽救0；保护集无长CV终局破坏；长hold有2挽救但新增1保护碰撞且裕度不足 |
| 已证实 | 长历史改善隐藏占据误差，但没有达到本轮主挽救门槛；投影接口本身可改变甚至破坏成功终局 |
| 证据支持的资源判断 | 当前冻结消费者/4s CV用途不值得继续算法化；不追加模型训练或参数搜索 |
| 尚未证明 | 所有PaS长期记忆方法无效；这8例失败都由遮挡造成；GRU缺少某个普遍能力；KDA在其他任务中没有价值 |

**唯一下一步：提交并归档本轮，停止这一入口。** 不自动改模型、迁移母体或再找第六个消费者。未来任何重启必须提出与这里不同的具体科学能力和独立证据，不能只以“还有没试过的模型”作为依据。

<!-- END PRESERVED SOURCE -->


---

<a id="stage-4"></a>

## Source: interaction-response-problem-audit.md

Original full-source SHA-256: c7cec34e71b8daf05ada790b043cf4659bd1929f56c746899226a7ba91eb7257

<!-- BEGIN PRESERVED SOURCE -->
# Interaction-Response Temporal Problem Audit

日期：2026-10-05。范围：只查SICNav/TRACER这两条既有线索，不训练、不修改PaS、不追加KDA、不开发新消费者。

## 1. 结论

**本轮没有找到可以进入新时序算法开发的、已验证动作收益的原生交互响应问题。暂停当前“新时序模型做导航”的模型研发，不自动扩搜索。**

不能把三件事混在一起：

1. 人会响应机器人：SICNav源码与本轮函数测试均确认。
2. 响应还依赖当前状态之外的持续交互状态：当前SICNav默认ORCA没有发现这个能力；TRACER以作者定义的持续模式协议研究过它，但不是我们的本地验证。
3. 合法历史能改善真实导航动作，并留下新方法空间：本轮没有这条证据。

上一轮PaS已封账：长CV独有挽救0/8、保护集8/8完成，长hold有不满足安全要求的挽救。保持pas-failure-confirmation.md的原结果。负结果支持停止具体用途，不能扩大为所有物理状态记忆或所有时序导航问题都已被否定。

## 2. 原生任务与资产核查

| 入口 | 本地/文献资产 | 任务变化 | 本轮判定 |
|---|---|---|---|
| SICNav / CrowdSimPlus | c702fb8源码、已安装sicnav环境、两个既有SICNav-np运行记录 | 机器人可见、原生人机响应、静态墙/瓶颈、unicycle MPC；不是shixu的80动作IL-MC | 即时交互可运行；没有所需的持续响应模式证据 |
| SICNav-Diffusion | 同仓库有JMID/iMID权重 | 学习预测与MPC的另一消费接口 | 仅资产核查，不转开旧预测桥梁，不运行新权重实验 |
| TRACER | 原论文及公式/消融；未核得官方代码+环境+checkpoint可运行组合 | 多机器人、持续响应mode及执行证据更新 | 有先例线索，复现资产阻塞；不能当作本地新发现 |

SICNav官方库确实提供ORCA/SFM与交互规划，不把“有官方代码”当成额外历史能力已经验证。[官方仓库](https://github.com/sepsamavi/safe-interactive-crowdnav)

本地temp副本已有他人/旧任务对CrowdSimPlus的修改：square生成器补env引用，discomfort_dist改读reward字典。未回退、未覆盖；这次使用hallway/circle，不新造噪声、遮挡或行为模式。协议记录脏状态、相关源码和旧资产哈希。

## 3. SICNav具体缺不缺跨时刻响应状态

以下是本地源码事实，而不是“理论上应该需要记忆”：

- crowd_sim_plus/envs/crowd_sim_plus.py:1043–1054：给每个人当前其他人的观测；robot.visible为true时，加入当前机器人位置、速度、半径。
- crowd_sim_plus/envs/utils/human_plus.py:105起：Human.act将自己的FullState、当前邻居、静态墙交给policy。自己的目标和v_pref仍参与环境生成，未作为新增可部署输入。
- crowd_sim_plus/envs/policy/orca_plus.py:55–71：以当前p/v/r、目标、v_pref构建ORCA并计算当前preferred velocity。
- 同文件:84–90：完成doStep后只记录last_state，并将sim置None；last_state没有在下一次响应计算中读取。没有“刚才被这个机器人逼近/让行后，持续改变某个响应参数”的更新。
- SFM备选: social_force.py:39起：当前p/v、目标与固定力参数决定更新；is_bottleneck来自场景，不是由执行交互学到的响应mode。本轮没有执行SFM对照。
- sicnav/policy/campc.py:1285–1315：非特权臂用当前p+2v估计目标。求解器warm start/上一解缓存是规划器状态，不是人的持续响应状态。

因此，在固定人目标、v_pref、邻居、墙和模型参数的完整当前状态下，默认ORCA响应不额外读取过去交互。**这不意味着机器人合法当前观测充分**：人的目标/v_pref仍可能隐藏。那属于另一种隐状态恢复问题，不能重新包装成已发现的持续让行响应记忆。

## 4. 有限回放与kill test

仅复用两个已有成功运行的SICNav-np case0记录，没有新episode、MPC solve或新导航策略：

| 既有配置 | 日志控制步 | 预先固定root tick | 当前物理状态最大回放偏差 |
|---|---:|---|---:|
| hallway_bottleneck，3人 | 15 | 0、7、14 | 1.11e-16 |
| circle_crossing，5人 | 18 | 0、9、17 | 1.85e-13 |

旧pkl包含终局摘要及优化x/u，不含完整原观测、真正执行命令、原奖励或响应belief。用保存u的首列，按原ActionRot的omega乘dt转换回放；每步将重建p/v/heading/signed-speed与保存x的当前列核对，容差事先固定1e-5。33步全部通过。

这提供有限的物理状态回放，不等于原完整训练/评估复现。保存x中未来列与估计目标没有冒充真实后果标签；两个成功记录也没有被说成自然失败cohort。

kill test：在6个预定root上，对每个人保持相同完整当前状态、目标、邻居、机器人及墙，比较已承接原生运行的ORCAPlus与相同参数的全新ORCAPlus。只重置人的policy缓存，不改变世界或消费者。

| 测试 | 结果 | 能说明什么 |
|---|---|---|
| 24次warm vs fresh响应查询 | **0/24改变，输出逐元素精确一致** | 当前默认ORCA缓存没有额外响应记忆作用，支持源码检查 |
| 函数正控制：只改变当前机器人速度 | **10/24改变** | 人确实消费当前机器人运动，不是“完全无交互” |
| 自然同几何、异历史、异响应失败配对 | **没有建立** | 不能宣布已找到时序决策问题 |
| 导航动作/Q/终局改善 | **没有测得/没有声称** | 上述是human-response函数测试，不是方法收益 |

正控制仅在函数输入中将当前robot velocity设为沿朝向的正/负v_pref；未执行这些速度，未修改benchmark可见性，也不是机器人动作最优性测试。

没有用近邻配对制造历史优势，也没有隐瞒当前goal/v_pref的差异：这些量在warm/cold比较中完全相同。这个kill test否证的是**默认ORCA中额外跨时刻响应缓存**，不是“人的真实社会行为没有历史”。

## 5. TRACER近邻核查与反证

TRACER III-C以执行动作和同步观察更新identity-bound模式belief；IV-A的持续模式试验由作者设置固定跨窗口mode，IV-C/D报告其预测及决策作用。这说明有相关研究先例，不是本地原生问题确认。未核得可直接运行的官方资产组合，不为复现其结论自行注入mode或重建系统。[TRACER原论文](https://arxiv.org/html/2609.18776v1)

**其外部SocialGym2结果不支持把完整系统收益归因于持续belief：** Table III(b)的完整TRACER完成率31.6%、碰撞率30.3%；Reset为32.2%、29.7%；NoBelief为32.2%、29.4%。完整版本相对GoAlone改善，不等于相对相同系统的记忆消融改善。没有原始配对数据或区间，不能把小差异说成显著劣势；也不能将合成模式结果外推成外部任务中的稳定时序增量。[Table III(b)](https://arxiv.org/html/2609.18776v1#S4.T3)

这是对早前“该先例已支持持续交互历史带来可迁移导航价值”解释的收紧。通用累计响应belief和identity一致更新已有覆盖；相似工作不自动拒绝方向，但当前没有核出我们的具体新增能力和实际残差。

## 6. 当前状态—缺失变量—历史—动作链

| 链条 | SICNav默认ORCA | TRACER线索 |
|---|---|---|
| 当前合法状态不充分 | 目标/v_pref可能隐藏；未证明新的响应模式歧义 | 作者定义latent mode，属于其任务设定 |
| 缺失的交互响应变量z | 没有找到由过去robot interaction持续更新的z | 明确有模式belief，但已经是其方法 |
| 合法历史可恢复z | 本轮未建立；warm/cold没有响应差异 | 作者受控实验有报告，本地未复现 |
| 改变合理导航动作并改善后果 | 没有本地证据 | 外部Reset/NoBelief汇总未显示完整版本增量 |
| 简单方法剩余缺口 | 未建立，不预设GRU失败 | 普通离散Bayes/累计统计是强替代；无本地比较 |

**没有一条线完成本轮要求的四段链条，因此不进入Top3或算法阶段。** 不把资产阻塞说成科学问题被否定，也不因“人机交互常见”自动升级。

## 7. 决策与重启条件

已证实：所查SICNav原生支持当前人机响应；24次固定完整状态的缓存重置没有改变人的响应；旧33步物理回放通过；TRACER已包含相关机制，其外部记忆消融不支持稳定增量主张。

有证据支持的判断：在这些既有资产上继续迁移KDA/GRU或人为注入社会响应mode，没有充分研发依据。暂停当前新时序模型路线。

尚未证明：真实人机历史一般无价值；其他原生交互任务不存在持续模式；没有人能在同类问题形成新贡献。这些不能由本轮结果推断。

唯一下一步：**归档后停止模型投入和自动搜索。** 重启至少需要原生任务的可运行持续响应资产，以及“当前信息匹配、合法执行历史能区分响应、真实动作后果改善”的证据；不要求GRU先失败才承认问题存在。还须单独说明相对既有累计belief/简单滤波的具体研究空间。

## 8. 复现资产

入口：experiments/interaction_response_audit.py。使用现有sicnav环境；没有改母体、权重、奖励、训练或规划器。

```bash
/home/abc/miniconda3/envs/sicnav/bin/python -m experiments.interaction_response_audit
```

脚本拒绝覆盖protocol/results。协议保存两份旧运行、9份母体代码/配置及诊断源码哈希，运行结束重新核查未变。首次临时探针缺少官方policy_factory注册导致KeyError，未产生结果；补正确官方导入后运行，未修改源码或dependency。

protocol SHA256：830fb3307c001d9b4272478c1b2483a642b63c9faa07be20f3c304b5d4563a63。

results SHA256：5e11a40d5007714d81a01aa77738097eac889f523052d9fe0db695776b40da36。归档前再次核对协议、结果、诊断源码、9份母体代码/配置和两份旧运行记录的哈希，全部一致。协议与24条原始查询结果一并纳入Git；外部依赖和旧pkl仍保留本地，不声称单独clone shixu即可完整复现。

结果目录：

/home/abc/workspace/shixu/outputs/interaction_response_audit/

新测试核对current列与未来/goal列隔离、signed speed、冻结结果禁止覆盖。所有本轮数字来自results.json；没有新导航checkpoint和收益报告。

完整仓库回归：181 passed，3 skipped，68 warnings；三个legacy parity测试因未设置原源码/checkpoint路径而跳过，警告来自现有NumPy/SciPy兼容性。执行命令为env PYTHONPATH=vendor:. /usr/bin/python -m pytest -q -rs tests。本轮未修改模型、训练、动作执行或母体源代码。

<!-- END PRESERVED SOURCE -->
