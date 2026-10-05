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
