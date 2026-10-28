from __future__ import annotations

from pathlib import Path

# Create 3 sample files sharing one IP (10.0.0.5) for corroboration demo
# Netflow sample
netflow_data = """timestamp,src_ip,dest_ip,flow_duration,dest_port,label
2023-01-01 00:00:01,10.0.0.5,10.0.0.80,30.5,80,BENIGN
2023-01-01 00:00:02,10.0.0.5,10.0.0.80,45.2,22,BENIGN
2023-01-01 00:00:03,10.0.0.5,10.0.0.81,25.3,21,DoS
2023-01-01 00:00:04,10.0.0.5,10.0.0.81,35.7,21,PortScan
2023-01-01 00:00:05,10.0.0.5,10.0.0.82,50.1,443,BENIGN
"""

# SIEM/EDR sample
siem_data = """timestamp,host_name,event_id,user_name,src_ip,label
2023-01-01 00:00:01,WS01,EV001,admin,10.0.0.5,BENIGN
2023-01-01 00:00:02,WS02,EV002,admin,10.0.0.5,Lateral Movement
2023-01-01 00:00:03,WS03,EV003,admin,10.0.0.5,Persistence
"""

# Cloud entity sample
cloud_data = """timestamp,principal_name,event_id,user_name,src_ip,label
2023-01-01 00:00:01,PV01,EV001,admin,10.0.0.5,BENIGN
2023-01-01 00:00:02,PV02,EV002,admin,10.0.0.5,Initial Access
2023-01-01 00:00:03,PV03,EV003,admin,10.0.0.5,Command and Control
"""

# Write files
data_dir = Path("data/raw")
(data_dir / "netflow").mkdir(parents=True, exist_ok=True)
(data_dir / "siem_edr").mkdir(parents=True, exist_ok=True)
(data_dir / "cloud_entity").mkdir(parents=True, exist_ok=True)

with open(data_dir / "netflow" / "sample1.csv", "w") as f:
    f.write(netflow_data)

with open(data_dir / "siem_edr" / "sample1.csv", "w") as f:
    f.write(siem_data)

with open(data_dir / "cloud_entity" / "sample1.csv", "w") as f:
    f.write(cloud_data)

print("Demo data created successfully!")
print("Files created:")
print("  data/raw/netflow/sample1.csv")
print("  data/raw/siem_edr/sample1.csv")
print("  data/raw/cloud_entity/sample1.csv")
