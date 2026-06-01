import datetime
from parameter import PERCEPTION_ALPHA
from component import Station, Truck


class PerceptionRecord:
    """单次调度或重定位的感知记录"""

    def __init__(self, time: datetime.datetime):
        self.time = time

    def hours_since(self, current_time: datetime.datetime) -> float:
        return (current_time - self.time).total_seconds() / 3600.0


class DispatchPerceptionRecord(PerceptionRecord):
    """调度感知记录"""

    def __init__(self, time, intent_sid, actual_sid, station_truck_num,
                 station_returned_count):
        super().__init__(time)
        self.intent_sid = intent_sid
        self.actual_sid = actual_sid
        self.station_truck_num = station_truck_num
        self.station_returned_count = station_returned_count

    @property
    def intent_matched(self):
        return self.intent_sid == self.actual_sid


class RepositionPerceptionRecord(PerceptionRecord):
    """重定位感知记录"""

    def __init__(self, time, from_pid, chosen_sid, available_sids):
        super().__init__(time)
        self.from_pid = from_pid
        self.chosen_sid = chosen_sid
        self.available_sids = available_sids
        self.received_positive = False  # 是否已获得正向奖励


class PerceptionLayer:
    """感知层：维护dispatch和reposition的历史记录，为RL提供奖励信号"""

    def __init__(self):
        self.dispatch_records: list[DispatchPerceptionRecord] = []
        self.reposition_records: list[RepositionPerceptionRecord] = []
        self.alpha = PERCEPTION_ALPHA

    def reset(self):
        self.dispatch_records.clear()
        self.reposition_records.clear()

    def record_dispatch(self, time, intent_sid, actual_sid, station):
        """记录一次调度事件"""
        record = DispatchPerceptionRecord(
            time=time,
            intent_sid=intent_sid,
            actual_sid=actual_sid,
            station_truck_num=station.truck_num,
            station_returned_count=station.returned_truck_count,
        )
        self.dispatch_records.append(record)
        return record

    def record_reposition(self, time, from_pid, chosen_sid, available_sids):
        """记录一次重定位事件"""
        record = RepositionPerceptionRecord(
            time=time,
            from_pid=from_pid,
            chosen_sid=chosen_sid,
            available_sids=available_sids,
        )
        self.reposition_records.append(record)
        return record

    def process_dispatch_result(self, env, dispatch_record: DispatchPerceptionRecord):
        """
        处理调度结果，向之前的Reposition记录写入奖励。
        返回本次分配到的正/负奖励，供RL Agent使用。

        当dispatch意图与实际不符时，向之前的Reposition序列写入负奖励。
        当dispatch意图与实际相符时，如果车来源于之前的Reposition，写入正奖励。
        """
        positive_rewards = []  # (reposition_record_index, reward)
        negative_rewards = []  # (reposition_record_index, reward)

        current_time = dispatch_record.time

        if not dispatch_record.intent_matched:
            # 意图不符：向之前的Reposition写入负奖励
            base_neg_reward = -1.0
            for idx, repo_record in enumerate(self.reposition_records):
                if repo_record.received_positive:
                    continue
                hours = repo_record.hours_since(current_time)
                scaled_reward = base_neg_reward * (self.alpha ** max(hours, 0))
                negative_rewards.append((idx, scaled_reward))
        else:
            # 意图相符：检查发出的车是否来源于之前的Reposition
            station = env.stations.get(dispatch_record.actual_sid)
            if station is not None and station.dispatch_came_from_reposition():
                # 找到最早Reposition到该Station的record
                for idx, repo_record in enumerate(self.reposition_records):
                    if repo_record.chosen_sid == dispatch_record.actual_sid and not repo_record.received_positive:
                        positive_rewards.append((idx, 1.0))
                        repo_record.received_positive = True
                        break

        return positive_rewards, negative_rewards

    def get_station_snapshot(self, station: Station) -> dict:
        """获取厂站当前状态的快照，供RL使用"""
        return {
            'sid': station.sid,
            'truck_num': station.truck_num,
            'returned_truck_count': station.returned_truck_count,
            'dispatch_recent_60min': station.count_dispatch_given_range(None, 60),
        }

    def get_last_reposition_record(self) -> RepositionPerceptionRecord | None:
        if self.reposition_records:
            return self.reposition_records[-1]
        return None
