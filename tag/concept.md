# Smart Tag Concept - Diet & Hunting in Near Real Time

**Sensors**
- IMU (accelerometer/gyroscope) ? detects bursts/attacks.
- Pressure + temperature ? depth / thermocline context.
- Narrow-band hydrophone ? chewing and prey sounds.
- Gastric proxy (bioimpedance or lightweight internal thermistor) ? ingestion events.

**Edge AI / Duty Cycle**
- Lightweight classifier flags potential feeding events (IMU trigger), activates the hydrophone for short windows, and logs the gastric proxy.
- Only **events** are transmitted, saving energy.

**Telemetry & Power**
- Local buffer; uplink via Argos/Iridium when the shark surfaces.
- Aggressive duty cycling; consider micro solar panels and firmware optimizations.

**Payload (compact)**
- Timestamp, latitude/longitude, depth, feeding-event probability, summary of extracted features.
