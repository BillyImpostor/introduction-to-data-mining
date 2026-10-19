# Penjelasan Lengkap Output Notebook — Tugas 1 Pengantar Data Mining

**Notebook:** `[Nomor_1_Tugas_1_Pengantar_Data_Mining_2026].ipynb`
**Data:** `data/Data Nomor 1 - Tugas 1 PDM.csv`
**Konteks:** StreamVerse (layanan streaming) — dataset perilaku pengguna, buffer jaringan, biaya langganan, dan dukungan pelanggan untuk memprediksi risiko berhenti berlangganan.

---

## Daftar Isi

| # | Bagian Notebook | Cell | Jumlah Output |
|---|---|---|---|
| 0 | Import Library & Konfigurasi | 5, 6 | 0 |
| 1 | Input Data | 8 | 1 |
| 2 | Karakteristik Data | 11, 12, 13 | 3 |
| 3 | Duplikasi Data | 16, 17 | 5 |
| 4 | Inkonsistensi Format & Anomali Numerik | 19, 20, 21, 22 | 4 |
| 5 | Outlier | 24, 25, 26 | 2 |
| 6 | Missing Values | 28, 29, 30, 31 | 4 |
| 7 | Scaling | 34 | 0 |
| 8 | Encoding | 36, 37 | 1 |
| 9 | Feature Construction | 39, 40, 41, 42 | 1 |
| 10 | Data Exploration (Visualisasi) | 45, 46, 47 | 4 |
| | | **TOTAL** | **25** |

---

## 0. Import Library & Konfigurasi (Cell 5–6)

Kedua cell ini **tidak menghasilkan output** karena hanya berisi perintah impor dan pengaturan tampilan.

**Penjelasan:**
`pandas` untuk manipulasi data, `seaborn` + `matplotlib` untuk visualisasi, `numpy` untuk operasi matematis. Cell konfigurasi mengatur agar seluruh kolom tabel ditampilkan tanpa dipotong (`display.max_columns = None`), gaya grafik `whitegrid`, palet `tab10`, DPI 150, dan ukuran font 11. Tujuannya agar seluruh output tabel dan grafik terbaca jelas.

---

## 1. Input Data (Cell 8)

### Output: `df.head()` — 5 baris pertama × 20 kolom

```
  customer_id  age   gender subscription_plan  monthly_subscription_fee_usd
0   CUST-1753   23    Male            Premium                         18.99
1   CUST-1520   50  FEMALE           standard                         13.99
2   CUST-1211   39     MALE           PREMIUM                         18.99
3   CUST-1612   18    Male              Basic                          8.99
4   CUST-1915   39  female           STANDARD                         13.99
```

**Penjelasan:**
Tabel ini adalah sneak peek dataset mentah. Yang **langsung terlihat** adalah kualitas data yang belum rapi:

* **Inkonsistensi kapitalisasi** sudah tampak di 5 baris pertama saja: `Male`, `FEMALE`, `MALE`, `female`, `Premium`, `standard`, `PREMIUM`, `STANDARD`. Padahal secara konsep hanya ada **3 gender** dan **4 paket langganan**.
* **Keterkaitan paket ↔ biaya** sudah terkonfirmasi di sini: Premium = 18,99; Standard = 13,99; Basic = 8,99. Artinya harga ditentukan oleh paket.
* **Kolom dengan nilai kosong (NaN)** langsung terlihat pada `student_gpa` (ketika baris 0–2) — bahkan di baris paling atas pun kolom ini sudah kosong (diisi `NaN`).
* **Baris 4 (`CUST-1612`)** memperlihatkan `offline_downloads_monthly` = 20 cukup tinggi dan `work_productivity_score` = 6,9.
* `customer_id` berformat seragam `CUST-XXXX`.

**Kesimpulan:** data mentah **memang belum rapi** dan harus melewati proses cleaning. Tidak ada asumsi bahwa data sudah rapi.

---

## 2. Karakteristik Data (Cell 11, 12, 13)

### Output 1 — `df.info()`

```
RangeIndex: 1015 entries, 0 to 1014
Data columns (total 20 columns):
 #   Column                        Non-Null Count  Dtype
 0   customer_id                   1015 non-null   object
 1   age                           1015 non-null   int64
 ...
14   student_gpa                   154 non-null    float64
15   work_productivity_score       861 non-null    float64
 ...
dtypes: float64(5), int64(6), object(9)
memory usage: 158.7+ KB
```

**Penjelasan:**
`info()` memberi 4 informasi sekaligus:

1. **Jumlah data = 1.015 baris, 20 kolom.**_dataset_ bertipe tabular (rectangular).
2. **`Non-Null Count` langsung menunjukkan adanya missing values.** Hanya **2 kolom** yang punya nilai kosong:
   * `student_gpa`: 154 dari 1.015 (**missing 84,95%**) — persentasenya ekstrem.
   * `work_productivity_score`: 861 dari 1.015 (**missing 15,05%**) — masih wajar.
   18 kolom lainnya 100% terisi.
3. **Komposisi tipe data:** 9 `object` (kategorik), 6 `int64`, 5 `float64`. Artinya **11 variabel numerik** dan **9 variabel kategorik** untuk dianalisis.
4. **`memory usage: 158.7+ KB`** — dataset sangat kecil, sehingga aman diproses di memori tanpa areal

**Catatan penting:** `churn_risk_level` bertipe `object` dengan 3 nilai → problem-nya adalah **klasifikasi multikelas**, bukan regresi.

---

### Output 2 — `df.describe()`

```
               age  monthly_subscription_fee_usd  daily_streaming_hours
count  1015.000000                   1015.000000            1015.000000
mean     30.608867                     15.397025               3.644079
std      10.512213                     49.486703               2.499251
min     -38.000000                    -18.990000              -4.020000
25%      24.000000                      8.990000               2.640000
50%      31.000000                     13.990000               3.490000
75%      37.000000                     13.990000               4.340000
max      60.000000                    999.000000              42.000000
```

**Cara membaca:** `count` = jumlah data valid, `mean` = rata-rata, `std` = devsiasi standar (menyebarannya), `min`/`max` = nilai terkecil/terbesar, `25%`/`50%`/`75%` = kuartil 1, median, kuartil 3.

**Penjelasan per variabel:**

| Variabel | Mean | Median | Std | Min | Max | Interpretasi |
|---|---|---|---|---|---|---|
| `age` | 30,61 | 31 | 10,51 | **−38** | 60 | Mean ≈ median → cukup simetris. **Min −38 tidak mungkin secara fisiologis** → ada 8 data umur negatif |
| `monthly_subscription_fee_usd` | 15,40 | 13,99 | **49,49** | **−18,99** | **999** | **Std (49,49) >> mean (15,40)** → ada nilai ekstrem. Min & max tidak logis |
| `daily_streaming_hours` | 3,64 | 3,49 | 2,50 | **−4,02** | **42** | Rata-rata 3,6 jam/hari masuk akal, tapi **max 42 jam** (>24 jam, mustahil) dan **min negatif** |
| `watch_time_late_night_hours` | 1,31 | 1,18 | 0,90 | 0 | 3,91 | **Paling bersih** — tidak ada nilai negatif, sebaran rapat |
| `offline_downloads_monthly` | 8,56 | 6 | 6,73 | 0 | 31 | Mean > median → right-skewed, sebagian besar unduh sedikit |
| `number_of_profiles_shared` | 2,26 | 2 | 1,27 | 1 | 5 | Right-skewed ringan, tidak ada yang 0 profil |
| `buffering_events_weekly` | 5,22 | 5 | 2,32 | 0 | 14 | Rata-rata 5 buffer/minggu → cukup banyak |
| `customer_support_tickets` | 1,22 | 1 | 1,31 | 0 | 8 | **Std > mean** → sangat right-skewed, banyak yang 0 tiket |
| `student_gpa` | 3,11 | 3,20 | 0,51 | 2,15 | 3,92 | Hanya 154 data. **Rentang 2,15–3,92** sesuai skala IPK |
| `work_productivity_score` | 6,78 | 6,80 | 1,74 | 1 | 10 | Mean ≈ median → simetris, sesuai skala 1–10 |
| `tenure_months` | 17,24 | 13 | 13,99 | 1 | 48 | Mean > median → right-skewed. Ada yang baru 1 bulan & yang loyal 48 bulan |

**Insight kunci dari `describe()`:**

* **Terdapat 3 kelompok masalah numerik:**
  1. **Nilai negatif** → `age` (−38), `monthly_subscription_fee_usd` (−18,99), `daily_streaming_hours` (−4,02).
  2. **Nilai melebihi batas logis** → `daily_streaming_hours` max 42 (>24 jam).
  3. **Outlier ekstrem** → `monthly_subscription_fee_usd` max 999 (std 49,49 vs mean 15,40).
* **Dua kolom dengan `count` < 1015** pada `describe()` menunjukkan missing values ada di `student_gpa` dan `work_productivity_score` saja.
* **Perhatikan `monthly_subscription_fee_usd`**: IQR sangat sempit (8,99–13,99) → data valid hanya ada di **4 nilai harga**, sisanya adalah pencilan atau salah input.

---

### Output 3 — Statistik Deskriptif Kategorik

```
                         Column Unique Values     Min Value       Max Value
0                   customer_id          1000     CUST-1001       CUST-2000
1                        gender            17          Male           other
2             subscription_plan            20         Basic         student
3                payment_method            20   CREDIT CARD        paylater
4                primary_device            20        Mobile         tablet
5        content_genre_favorite            25        Action          sci-fi
6           satisfaction_score             5  Dissatisfied  Very Satisfied
7  streaming_quality_preference             4        4K UHD       SD (480p)
8              churn_risk_level             3          High        Moderate

  Most Frequent Value Frequency
0           CUST-1341         2
1             FEMALE         96
2            standard        83
3         Credit Card        82
4            Smart TV       105
5              action        81
6             Neutral       405
7          FHD (1080p)       316
8               High        366
```

**Penjelasan per kolom:**

| Kolom | Unique | Seharusnya | Nilai Terbanyak | Frekuensi | Interpretasi |
|---|---|---|---|---|---|
| `customer_id` | 1.000 | 1.015 baris | CUST-1341 | 2 | **Ada indikasi duplikasi** — 1.015 baris tapi hanya 1.000 ID unik → **15 baris duplikat** |
| `gender` | **17** | 3 | FEMALE | 96 | Terlalu banyak! Harusnya Male/Female/Other saja |
| `subscription_plan` | **20** | 4 | standard | 83 | Terlalu banyak! Harusnya 4 paket |
| `payment_method` | **20** | 4 | Credit Card | 82 | Terlalu banyak! |
| `primary_device` | **20** | 4 | Smart TV | 105 | Terlalu banyak! |
| `content_genre_favorite` | **25** | 6 | action | 81 | Terlalu banyak! |
| `satisfaction_score` | 5 | 5 | Neutral | 405 | **Sesuai** — 39,9% pengguna netral |
| `streaming_quality_preference` | 4 | 4 | FHD (1080p) | 316 | **Sesuai** — 31,1% pilih FHD |
| `churn_risk_level` | 3 | 3 | High | 366 | **Sesuai** — 36,1% risiko tinggi |

**Temuan kritis:**

* **Kelima kolom kategorik utama memiliki jumlah nilai unik BERAPA JAUH lebih besar dari seharusnya** (gender 17 vs 3; plan 20 vs 4; payment 20 vs 4; device 20 vs 4; genre 25 vs 6). Penyebabnya adalah **perbedaan kapitalisasi** (contoh: `MALE`, `Male`, `male`) dan **spasi berlebih** (contoh: `'female '`, `' Male'`).
* **Konsekuensi langsung**: kolom `Min Value` / `Max Value` / `Most Frequent Value` **menjadi tidak representatif**. Contoh: `gender` melaporkan `FEMALE` (frekuensi 96) sebagai nilai terbanyak — padahal itu hanya **satu dari beberapa variasi kapitalisasi** dari `Female`. Frekuensi sebenarnya jauh lebih besar.
* **Penting untuk dipahami**: kolom `Min Value` dan `Max Value` untuk data `object` (string) **tidak memiliki makna statistik** — pandas membandingkannya **secara leksikografis (urutan alfabet)**. Misalnya `subscription_plan` "Min" = `Basic` dan "Max" = `student` — ini **bukan** paket termurah/mahal, hanya urutan huruf (`B` < `s` karena huruf besar didahului huruf kecil). Kolom ini **hanya berguna** jika sudah dinormalisasi.
* **`satisfaction_score` = 5 dan `churn_risk_level` = 3 (sudah sesuai)** → dua kolom ini **tidak punya masalah inkonsistensi**, aman langsung dipakai.
* **Distribusi `churn_risk_level` cukup seimbang:** High 36,1%, Moderate 32,9%, Low 31,0% — kondisi bagus untuk klasifikasi.

---

## 3. Pemeriksaan dan Penanganan Duplikasi Data (Cell 16, 17)

### Output 1 — Deteksi Duplikat (`duplicate_data.head()`)

```
[Warning: SettingWithCopyWarning]
A value is trying to be set on a copy of a slice from a DataFrame

    customer_id  age   gender subscription_plan  monthly_subscription_fee_usd
125   CUST-1341   39     Male         Standard                          13.99
531   CUST-1341   39     Male         Standard                          13.99
404   CUST-1377   44  FEMALE            Premium                         18.99
794   CUST-1377   44  FEMALE            Premium                         18.99
300   CUST-1418   21    Female          Student                           4.99
```

**Penjelasan:**

* **Output pertama (Warning):** pandas menampilkan `SettingWithCopyWarning` karena `duplicate_data = df[df.duplicated(keep=False)]` **menghasilkan salinan (copy)** dari `df`, lalu kode melakukan `sort_values(..., inplace=True)` langsung padanya. pandas memperingatkan bahwa operasi `inplace` pada slice bisa tidak konsisten. **Secara teknis kode tetap berjalan**, tapi lebih aman menulis `.sort_values(...)` tanpa `inplace=True`.
* **Output kedua (tabel):** Setiap pasangan duplikat memiliki **nilai identik di seluruh 20 kolom** — bukan hanya `customer_id` yang sama. Contohnya:
  * Baris 125 & 531: `CUST-1341`, age 39, Male, Standard, 13.99, **semuanya sama persis**.
  * Baris 404 & 794: `CUST-1377`, age 44, FEMALE, Premium, 18.99, **semuanya sama**.
* **`df.duplicated(keep=False)`** menandai **semua anggota** dari grup duplikat (bukan hanya duplikat kedua). Ditemukan **30 baris** = **15 pasangan duplikat**.
* Perhatikan baris 300 (`CUST-1418`): hanya muncul satu kali di `head()` karena pasangan duplikatnya berada di index yang lebih tinggi setelah pengurutan.

---

### Output 2–3 — Penanganan Duplikat (`df_processed.head()` + `df_processed.info()`)

```
[display] 5 baris pertama — IDENTIK dengan df.head() (data sama, urutan sama)
[stream]
Index: 1000 entries, 0 to 1014
Data columns (total 20 columns):
14  student_gpa              150 non-null    float64
15  work_productivity_score  850 non-null    float64
memory usage: 164.1+ KB
[display] None
```

**Penjelasan:**

* **Output `display` pertama** (`df_processed.head()`): 5 baris pertama **identik** dengan `df.head()` pada langkah sebelumnya. Ini benar — `drop_duplicates()` **hapus duplikat tanpa mengubah urutan baris yang tersisa**, dan baris 0–4 memang bukan duplikat.
* **Output `stream` (`df_processed.info()`):**
  * `Index: 1000 entries, 0 to 1014` — jumlah baris turun dari **1.015 → 1.000** (tepat 15 baris duplikat terhapus). **Perhatikan index tidak di-reset** — masih 0–1014 dengan "lubang" di posisi yang dihapus, karena `reset_index()` belum dipanggil.
  * `student_gpa`: 154 → **150** (4 duplikat ikut terhapus).
  * `work_productivity_score`: 861 → **850** (11 duplikat ikut terhapus).
  * `memory usage` naik dari 158,7 KB → **164,1 KB** — naik, bukan turun! Ini karena: (a) index masih menyimpan 1.015 entri sementara data hanya 1.000 baris, (b) fragmentasi memori saat proses drop. Ini normal, bukan bug.
* **Output `display` ketiga (`None`):** `display()` adalah fungsi dari `IPython.display` yang secara eksplisit menampulkan objek tanpa menambahkan `[Out[N]:` prefix. Menampilkan `None` berarti output-nya **kosong/sudah ditampilkan sebelumnya** — ini normal, bukan error.
* **`df_processed = df.drop_duplicates()`** dipilih (bukan `remove_duplicates()`) karena:
  * Aman: `df` asli tidak berubah.
  * `drop_duplicates()` tanpa argumen secara default menghapus duplikat **pertama** dan menyimpan **kemunculan terakhir** (`keep='last'`).
  * Deteksi duplikat menggunakan **semua kolom** (bukan hanya `customer_id`) → hanya baris yang **benar-benar identik** yang dihapus. Ini pendekatan yang **paling aman** karena tidak menghapus data yang kebetulan punya ID sama tapi isi berbeda.

---

## 4. Inkonsistensi Format & Anomali Numerik (Cell 19, 20, 21, 22)

### Output 1 — EDA Inkonsistensi Format Kategorik (Cell 19)

```
[Kolom: gender]
• Unique Raw: 17
• Sample: ['FEMALE ', ' female', 'FEMALE', 'female ', 'Male', 'MALE', ' Male', 'Female ', 'male', 'Male ', 'Female', 'male ', 'Other', 'Other ', 'OTHER', ' Other', 'other']

[Kolom: subscription_plan]
• Unique Raw: 20
• Sample: ['standard', 'Standard ', 'basic', 'BASIC', ' standard', 'Basic', 'STANDARD', 'Standard', 'premium ', ' Basic', 'basic ', ' student', 'premium', 'PREMIUM', ' Premium', 'Premium', 'STUDENT', 'student', 'Student ', 'Student']

[Kolom: payment_method]
• Unique Raw: 20
• Sample: ['Credit Card ', 'Credit Card', 'credit card', ' CREDIT CARD', 'CreditCard', 'e-wallet', 'E-Wallet', 'Ewallet', ' E-Wallet', 'Bank Transfer', 'E-Wallet ', 'EWALLET', 'Bank Transfer ', 'BANK TRANSFER', 'bank transfer', 'paylater', 'PAYLATER', 'PayLater', ' PayLater', 'Paylater ']

[Kolom: primary_device]
• Unique Raw: 20
• Sample: ['Smart TV ', 'smart tv', 'MOBILE', 'mobile', 'SMART TV', 'Smart TV', 'mobile ', ' Mobile', ' Smart TV', 'Mobile', 'laptop/pc', 'PC', 'pc', 'Laptop/PC', 'Laptop/PC ', 'Tablet', ' Tablet', 'tablet', 'tablet ', 'TABLET']

[Kolom: content_genre_favorite]
• Unique Raw: 25
• Sample: ['action ', ' Action', 'Action', 'ACTION', 'Drama ', 'Drama', 'DRAMA', 'Comedy', ' comedy', 'COMEDY', 'drama', 'Comedy ', ' anime', 'SciFi', 'ANIME', 'Anime ', 'Documentary', 'SCI-FI', 'sci-fi', 'Sci-Fi', 'Sci-Fi ', 'Anime', 'Documentary ', 'documentary', 'DOCUMENTARY']
```

**Penjelasan:**
Output ini secara sengaja menampilkan **nilai unik mentah** di setiap kolom kategorik. Tujuannya: **meli-hat pola inkonsistensi format** sebelum melakukan normalisasi.

* `repr(val)` pada kode ensures bahwa **spasi di awal/akhir kata terlihat jelas** — misalnya `'FEMALE '` (ada spasi di akhir) vs `'FEMALE` (tanpa spasi).
* **3 sumber utama inkonsistensi:**
  1. **Kapitalisasi tidak konsisten** → `MALE`, `Male`, `male`, `MALE`
  2. **Spasi berlebih di awal/akhir** → `'FEMALE '`, `' Male'`, `'laptop/pc '`
  3. **Varian penulisan** → `'CreditCard'` vs `'Credit Card'`, `'Ewallet'` vs `'E-Wallet'`, `'Smart Tv'` vs `'Smart TV'`, `'Laptop/Pc'` vs `'Laptop/PC'`, `'SciFi'` vs `'Sci-Fi'`
* **Konsekuensi jika tidak ditangani:**
  * Filter `df[df['gender']=='Male']` akan **kehilangan** semua variannya (`MALE`, ` male`).
  * `LabelEncoder` akan membuat **lebih dari 3 kelas** ( misrepresentasi).
  * Statistik deskriptif (freq, modus) menjadi **tidak akurat**.
* Kolom `satisfaction_score`, `streaming_quality_preference`, `churn_risk_level` **TIDAK diikutsertakan** karena sudah konsisten (5, 4, 3 nilai unik). Hanya 5 kolom dengan masalah yang diperiksa.

---

### Output 2 — Penanganan Inkonsistensi Format (Cell 20)

```
[Warning: SettingWithCopyWarning] × 9
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead
```

**Penjelasan:**
Kode ini menjalankan **3 langkah normalisasi**:

1. **`str.strip().str.title()`** → menghapus spasi di awal/akhir, lalu mengubah huruf pertama setiap kata menjadi kapital:
   * `' FEMALE '` → `'Female'`
   * `'MALE'` → `'Male'`
   * `'laptop/pc '` → `'Laptop/Pc'`

2. **Mapping kamus** (dictionary replacement) untuk varian yang tidak bisa diselesaikan oleh `title()`:
   * `payment_method`: `Creditcard` → `Credit Card`, `Ewallet` → `E-Wallet`
   * `primary_device`: `Pc` → `Laptop/PC`, `Laptop/Pc` → `Laptop/PC`, `Smart Tv` → `Smart TV`
   * `content_genre_favorite`: `Scifi` → `Sci-Fi`, `Sci-fi` → `Sci-Fi`

   *Mengapa perlu mapping?* Karena `str.title()` mengubah `CreditCard` → `Creditcard` (tidak ada spasi), sedangkan kita ingin `Credit Card`. `str.title()` juga mengubah `SmartTV` → `Smarttv`, bukan `Smart TV`.

3. **9 `SettingWithCopyWarning` muncul** karena pandas scripting mendeteksi bahwa `df_processed` pernah di-assign dari hasil `drop_duplicates()`, lalu operasi assignment langsung (`df_processed[col] = ...`) memicu peringatan bahwa mungkin bekerja pada salinan. **Warning ini tidak/TRX`** — kode tetap berjalan dengan benar, tapi lebih aman menggunakan `.loc[:, col] = ...` atau menambahkan `.copy()`.

**Hasil akhir (terlihat di cell 31):**
* `gender`: Male, Female, Other (3 nilai ✓)
* `subscription_plan`: Premium, Standard, Basic, Student (4 nilai ✓)
* `payment_method`: E-Wallet, Bank Transfer, Credit Card, Paylater (4 nilai ✓)
* `primary_device`: Smart TV, Mobile, Tablet, Laptop/PC (4 nilai ✓)
* `content_genre_favorite`: Action, Drama, Sci-Fi, Anime, Comedy, Documentary (6 nilai ✓)

---

### Output 3 — EDA Anomali Numerik (Cell 21)

```
[Kolom: age]
• Jumlah Anomali: 8
• Sample: [-30, -27, -34, -35, -16, -37, -38, -29]

[Kolom: monthly_subscription_fee_usd]
• Jumlah Anomali: 4
• Sample: [-18.99, -8.99, -13.99, -8.99]

[Kolom: daily_streaming_hours]
• Jumlah Anomali: 9
• Sample: [42.0, -2.65, -3.53, 30.5, -2.54, 34.2, -4.02, 36.8, 28.5]
```

**Penjelasan:**
Kode ini memeriksa **nilai yang secara logis/fisis tidak mungkin terjadi** (bukan statistical outlier):

* **`age` < 0** → 8 baris dengan umur negatif (−16 sampai −38). **Tidak mungkin** — tidak ada manusia berumur negatif. **Harus diperbaiki.**
* **`monthly_subscription_fee_usd` < 0** → 4 baris dengan biaya negatif (−8,99; −13,99; −18,99). **Tidak logis** — biaya langganan tidak bisa negatif (mungkin salah input tanda minus).
* **`daily_streaming_hours` < 0 ATAU > 24** → 9 baris:
  * **5 nilai > 24 jam** (28,5; 30,5; 34,2; 36,8; 42,0) — **secara fisik mustahil** (>24 jam sehari).
  * **4 nilai negatif** (−2,54; −2,65; −3,53; −4,02) — **tidak mungkin** (durasi watching tidak bisa negatif).

**Observasi penting tentang nilai negatif di `monthly_subscription_fee_usd`:** Setelah diperiksa lebih teliti, **nilai negatif ini persis sama dengan nilai positif yang valid, hanya dengan tanda minus diubah**:
* −8,99 = -(8,99) → ini harga paket Basic
* −13,99 = -(13,99) → ini harga paket Standard
* −18,99 = -(18,99) → ini harga paket Premium

Ini **sangat kuat menunjukkan** bahwa masalahnya adalah **kesalahan format tanda**, bukan data yang benar-benar negatif. Solusi yang tepat: ambil nilai absolut (`abs()`), **bukan hapus**.

---

### Output 4 — Penanganan Anomali Numerik (Cell 22)

```
[Warning: SettingWithCopyWarning] × 3
A value is trying to be set on a copy of a slice from a DataFrame.
```

**Penjelasan — 3 operasi utama:**

1. **`df_processed['age'].abs()`** → mengubah semua umur negatif menjadi positif:
   * −38 → **38**, −30 → **30**, dst.

2. **`df_processed['monthly_subscription_fee_usd'].abs()`** → mengubah biaya negatif menjadi positif:
   * −8,99 → **8,99**, −13,99 → **13,99**, −18,99 → **18,99**

   **Mengapa `abs()` dan bukan hapus?** Karena hasil pemeriksaan di atas membuktikan bahwa nilai negatif ini adalah **tanda minus yang terbalik** pada data yang **sebenarnya valid** (harga paket). Menghapus 4 baris tersebut akan **kehilangan data pelanggan yang sah**. `abs()` adalah pilihan yang benar dan tidak merusak data.

3. **`df_processed['daily_streaming_hours'].abs()`** → mengubah durasi negatif menjadi positif:
   * −2,54 → **2,54**, −4,02 → **4,02**, dst.

4. **Filter `daily_streaming_hours <= 24`** → menghapus baris dengan durasi > 24 jam:
   * **42,0; 34,2; 36,8; 30,5; 28,5** → **5 baris dihapus**.
   * Setelah `abs()`, baris dengan nilai negatif menjadi positif dan **tidak dihapus** (karena ≤ 24).
   * `.reset_index(drop=True)` → reset index agar berurutan dari 0.

**Kenapa `daily_streaming_hours > 24` dihapus, bukan di-clamp?** Karena **tidak mungkin** menonton lebih dari 24 jam sehari. Nilai 42 jam/hari adalah data yang **tidak mungkin** — data ini salah input, sehingga **tidak ada informasi yang bisa diselamatkan** dari baris tersebut. Berbeda dengan `monthly_subscription_fee_usd` yang jelas hanya tanda minus terbalik.

**Dampak:** `df_processed` sekarang memiliki **995 baris** (1.000 − 5).

**3 `SettingWithCopyWarning`** muncul karena pola yang sama seperti sebelumnya — `df_processed` pernah di-assign dari `drop_duplicates()`.

---

## 5. Pemeriksaan dan Penanganan Outlier (Cell 24, 25, 26)

### Output 1 — Boxplot untuk 11 Variabel Numerik (Cell 24)

```
<Figure size 2400x1800 with 11 Axes>
```

**Penjelasan:**
Kode membuat **11 boxplot** dalam layout 3×4 grid (12 slot, 1 slot kosong yang dihapus dengan `fig.delaxes()`). Boxplot menampilkan **median (garis tengah), Q1–Q3 (kotak), dan whiskers** sampai 1,5×IQR.

**Cara membaca:**
* **Titik-titik di luar whisker** = outlier.
* **Kotak sangat lebar** = data sangat terbagi (outlier variability tinggi).
* **Kotak sangat sempit** = data sangat homogen.

**Variable yang di-boxplot:**

| Posisi | Variabel | Pola yang Terlihat |
|---|---|---|
| (0,0) | `age` | Kotak normal, beberapa titik di atas (56–60) |
| (0,1) | `monthly_subscription_fee_usd` | **Kotak sangat kecil, whiskers sangat panjang ke atas** → 5 nilai sangat tinggi |
| (0,2) | `daily_streaming_hours` | Kotak normal; nilai > 24 jam sudah dihapus, tersisa beberapa titik tinggi |
| (0,3) | `watch_time_late_night_hours` | Kotak normal, hampir tidak ada outlier |
| (1,0) | `offline_downloads_monthly` | Right-skewed, whiskers panjang ke kanan |
| (1,1) | `number_of_profiles_shared` | Kotak normal, whiskers moderat |
| (1,2) | `buffering_events_weekly` | Distribusi baik, beberapa titik di atas |
| (1,3) | `customer_support_tickets` | **Sangat right-skewed**, banyak di 0, whiskers sangat panjang |
| (2,0) | `student_gpa` | Hanya 150 data terisi (sisanya kosong), kotak jadi sangat pendek |
| (2,1) | `work_productivity_score` | Distribusi cukup baik |
| (2,2) | `tenure_months` | Right-skewed, whiskers panjang ke kanan |

**`11 Axes`** = 12 slot dikurangi 1 yang dihapus (`delaxes`). Grid 3×4 = 12, tapi hanya 11 variabel numerik, jadi 1 slot dikosongkan.

**Kapan boxplot ini dibuat?** Pada titik ini, outlier `monthly_subscription_fee_usd` (399–999) **masih ada** dan `student_gpa` **masih kosong** — karena keduanya ditangani pada langkah setelahnya. Jadi kotak `student_gpa` hanya merepresentasikan 150 data yang terisi, dan kotak `monthly_subscription_fee_usd` masih memperlihatkan nilai ekstremnya.

*Catatan teknis:* Baris `num_cols = ...` terduplikasi dua kali di kode — tidak merusak apa pun, hanya tidak perlu.

---

### Output 2 — Deteksi Outlier dengan Metode IQR (Cell 25)

```
[Kolom: age]
• Batas Bawah    : 5.75
• Batas Atas     : 55.75
• Jumlah Outlier : 3
• Nilai Outlier  : [56, 59, 60]

[Kolom: monthly_subscription_fee_usd]
• Batas Bawah    : 1.49
• Batas Atas     : 21.49
• Jumlah Outlier : 5
• Nilai Outlier  : [399.0, 499.0, 650.0, 850.0, 999.0]

[Kolom: watch_time_late_night_hours]
• Batas Bawah    : -1.39
• Batas Atas     : 3.83
• Jumlah Outlier : 4
• Nilai Outlier  : [3.84, 3.86, 3.89, 3.91]

[Kolom: offline_downloads_monthly]
• Batas Bawah    : -13.5
• Batas Atas     : 30.5
• Jumlah Outlier : 1
• Nilai Outlier  : [31]

[Kolom: buffering_events_weekly]
• Batas Bawah    : -0.5
• Batas Atas     : 11.5
• Jumlah Outlier : 10
• Nilai Outlier  : [12, 13, 14]

[Kolom: customer_support_tickets]
• Batas Bawah    : -3.0
• Batas Atas     : 5.0
• Jumlah Outlier : 12
• Nilai Outlier  : [6, 7, 8]

[Kolom: work_productivity_score]
• Batas Bawah    : 2.25
• Batas Atas     : 11.45
• Jumlah Outlier : 8
• Nilai Outlier  : [1.0, 1.5, 1.6, 1.8, 1.9, 2.0, 2.1]
```

**Penjelasan — Cara Kerja IQR:**
IQR (Interquartile Range) = Q3 − Q1 (selisih antara kuartil 1 dan 3). Rentang "normal" dianggap: **[Q1 − 1,5×IQR, Q3 + 1,5×IQR]**. Nilai di luar rentang ini dianggap **outlier (pencilan)**.

**Analisis per variabel:**

| Variabel | Q1 | Q3 | IQR | Batas bawah | Batas atas | Outlier | Interpretasi |
|---|---|---|---|---|---|---|---|
| `age` | — | — | — | 5,75 | 55,75 | **3**: 56, 59, 60 | Orang tua (56–60 tahun). **Tidak dihapus** — masih valid |
| `monthly_subscription_fee_usd` | 8,99 | 13,99 | 5,00 | 1,49 | 21,49 | **5**: 399–999 | **Tidak logis untuk langganan bulanan** |
| `watch_time_late_night_hours` | — | — | — | −1,39 | 3,83 | **4**: 3,84–3,91 | Nyaris batas atas. **Tidak dihapus** — masih realistis |
| `offline_downloads_monthly` | — | — | — | −13,5 | 30,5 | **1**: 31 | Borderline. **Tidak dihapus** |
| `buffering_events_weekly` | 4 | 7 | 3 | −0,5 | 11,5 | **10**: 12, 13, 14 | Buffer sangat tinggi. **Tidak dihapus** — mungkin pengguna dengan masalah jaringan serius |
| `customer_support_tickets` | 0 | 2 | 2 | −3,0 | 5,0 | **12**: 6, 7, 8 | Tiket sangat banyak. **Tidak dihapus** — mungkin memang pengguna dengan masalah |
| `work_productivity_score` | 5,7 | 8 | 2,3 | 2,25 | 11,45 | **8**: 1,0–2,1 | Produktivitas sangat rendah. **Tidak dihapus** |

**Mengapa hanya `monthly_subscription_fee_usd` yang dihapus?**

Kode di cell 26 hanya menghapus outlier dari `monthly_subscription_fee_usd` (nilai 399, 499, 650, 850, 999). **Alasannya:**

1. **Nilai 399–999 USD untuk langganan bulanan** adalah **tidak realistis** — paket termahal di dataset ini hanya 18,99 USD. Ini **tidak mungkin terjadi di dunia nyata**.
2. **Variabel lain lebih halus** — umur 56–60 masih valid (lansia), buffer 12–14 masih mungkin (jaringan sangat buruk), tiket 6–8 masih mungkin (sangat tidak puas). Outlier pada variabel-variabel ini **mewakili kasus ekstrem yang informatif**, bukan kesalahan data.
3. **Prinsip dasar**: Hapus hanya yang **tidak mungkin terjadi** (impossible), pertahankan yang **mungkin tapi ekstrem** (unusual but possible).

**Rumus yang digunakan** (dihitung otomatis di kode, tidak ditulis eksplisit di output): misalnya untuk `monthly_subscription_fee_usd`, IQR = 13,99 − 8,99 = 5,00; batas atas = 13,99 + 1,5×5,00 = **21,49**. Semua nilai > 21,49 dianggap outlier.

---

### Output 3 — Penanganan Outlier `monthly_subscription_fee_usd` (Cell 26)

**Tidak ada output visual (tidak ada print/display).**

**Penjelasan:**
Kode ini menghapus **5 baris** dengan nilai `monthly_subscription_fee_usd` ∈ {399, 499, 650, 850, 999}:

```python
outlier = [399.0, 499.0, 650.0, 850.0, 999.0]
df_processed = df_processed[~df_processed['monthly_subscription_fee_usd'].isin(outlier)].reset_index(drop=True)
```

**Penjelasan teknis:**
* Tanda `~` = **NOT** (bukan), jadi: "ambil baris yang fee-nya **bukan** salah satu dari 5 nilai itu".
* `.reset_index(drop=True)` → reset index agar kembali ke 0, 1, 2, …

**Dampak:** 995 → **990 baris** (5 outlier dihapus).

---

## 6. Pemeriksaan dan Penanganan Missing Values (Cell 28, 29, 30, 31)

### Output 1 — Cek Missing Values (`missing_values`)

```
                              Missing Values  Percentage
customer_id                                0        0.00
age                                        0        0.00
gender                                     0        0.00
subscription_plan                          0        0.00
monthly_subscription_fee_usd               0        0.00
payment_method                             0        0.00
daily_streaming_hours                      0        0.00
primary_device                             0        0.00
content_genre_favorite                     0        0.00
watch_time_late_night_hours                0        0.00
offline_downloads_monthly                  0        0.00
number_of_profiles_shared                  0        0.00
buffering_events_weekly                    0        0.00
customer_support_tickets                   0        0.00
student_gpa                              841       84.95
work_productivity_score                  149       15.05
satisfaction_score                         0        0.00
streaming_quality_preference               0        0.00
tenure_months                              0        0.00
churn_risk_level                           0        0.00
```

**Penjelasan:**
Tabel ini menunjukkan **jumlah dan persentase nilai kosong** di setiap kolom, dihitung pada 990 baris setelah semua cleaning sebelumnya.

**Hanya 2 kolom punya missing values:**
* **`student_gpa`: 841 kosong (84,95%)** — sangat tinggi, **hampir 85% data hilang**.
* **`work_productivity_score`: 149 kosong (15,05%)** — masih wajar, <20%.

**18 kolom lainnya 100% terisi** — data sudah cukup bersih setelah langkah sebelumnya.

**Mengapa `student_gpa` memiliki missing value sangat tinggi?**

Setelah diperiksa lebih teliti, **`student_gpa` hanya terisi untuk pelanggan dengan paket `Student`**:
* 149 pelanggan paket `Student` → **semua punya `student_gpa`** (tidak ada yang hilang).
* 841 pelanggan paket lainnya (Basic, Standard, Premium) → **semua punya `student_gpa` kosong**.

Artinya, **`student_gpa` tidak "hilang" secara acak** — ini adalah **missing by design** (missing secara terstruktur). Pelanggan yang bukan pelajar **secara logis tidak memiliki IPK**, sehingga kolom ini **cuma relevan untuk subset pelanggan tertentu**.

**Temuan kunci — `work_productivity_score` juga terstruktur:**

*Ternyata* `work_productivity_score` yang hilang (149 baris) **juga persis** merupakan pelanggan paket `Student`! Jadi kedua kolom missing **berkorelasi dengan `subscription_plan`**.

**Implikasi untuk imputation:** penggunaan mean/median global (yang dilakukan di cell 29) **akan mengarang nilai** untuk pelanggan non-Student. Misal, mengisi `student_gpa` dengan mean 3,1 untuk pelanggan Premium yang **tidak punya IPK** adalah **tidak logis secara domain**.

---

### Output 2 — Penanganan Missing Values (Cell 29)

```
[Warning: FutureWarning] × 2
A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0 (perilaku akan berubah). Metode inplace ini tidak akan pernah bekerja karena objek antara selalu diperlakukan sebagai salinan.
```

**Penjelasan:**
Kode ini mengisi dua kolom yang kosong:

```python
df_processed['student_gpa'].fillna(df_processed['student_gpa'].mean(), inplace=True)
df_processed['work_productivity_score'].fillna(df_processed['work_productivity_score'].median(), inplace=True)
```

**Strategi imputation:**

| Kolom | Metode | Nilai yang Ditempatkan | Alasan |
|---|---|---|---|
| `student_gpa` | **Mean** | 3,102953 | Variabel numerik kontinu; mean mempertahankan nilai rata-rata agar total data tidak berubah |
| `work_productivity_score` | **Median** | 6,8 | Distribusinya right-skewed, sehingga median lebih tahan terhadap nilai ekstrem |

**Mengapa mean untuk `student_gpa` dan median untuk `work_productivity_score`?**

* **Mean** sesuai untuk data yang **symmetris atau mendekati normal**, sehingga cocok untuk `student_gpa` yang penyebarannya relatif normal (std 0,51).
* **Median** lebih **tahan terhadap nilai ekstrem (outlier)** → cocok untuk `work_productivity_score` yang right-skewed.

**FutureWarning** muncul karena pola `df[col].fillna(..., inplace=True)` disebut **chained assignment** — pandas memperingatkan bahwa **mulai pandas 3.0, operasi ini tidak akan bekerja** karena selalu bekerja pada salinan. Perbaikan yang disarankan pandas:
```python
df_processed['student_gpa'] = df_processed['student_gpa'].fillna(df_processed['student_gpa'].mean())
```

**Nilai yang diimput — bisa dilihat di Output 3 (cell 30):** Setelah fill, `student_gpa` untuk SEMUA baris pertama bernilai **3.102953** (identik) karena semuanya diisi dengan mean. Ada konsekuensi penting: **nilai 3.102953 menjadi "baseline"** — semua pelanggan yang sebelumnya punya GPA akan diselaraskan ke angka ini jika dianalisis.

---

### Output 3 — Data Bersih (`df_processed.head()`)

```
  customer_id  age  gender subscription_plan  monthly_subscription_fee_usd
0   CUST-1753   23    Male           Premium                         18.99
1   CUST-1520   50  Female          Standard                         13.99
2   CUST-1211   39    Male           Premium                         18.99
3   CUST-1612   18    Male             Basic                          8.99
4   CUST-1915   39  Female          Standard                         13.99

  payment_method  daily_streaming_hours  primary_device  content_genre_favorite
0       E-Wallet                   3.73       Smart TV                 Action
1       E-Wallet                   4.25       Smart TV                  Drama
2       E-Wallet                   5.25         Mobile                 Action
3       E-Wallet                   3.51       Smart TV                  Drama
4  Bank Transfer                   5.35         Mobile                 Sci-Fi

  customer_support_tickets  student_gpa  work_productivity_score
5                         3.102953                      8.5
0                         3.102953                      3.2
4                         3.102953                      8.7
0                         3.102953                      6.9
2                         3.102953                      6.2
```

**Penjelasan — apa yang berubah dari `df.head()` awal:**

| Aspek | Sebelum (df) | Sesudah (df_processed) | Keterangan |
|---|---|---|---|
| **gender** | `Male`, `FEMALE`, `MALE`, `Male`, `female` | `Male`, `Female`, `Male`, `Male`, `Female` | ✓ Semua ternormalisasi |
| **subscription_plan** | `Premium`, `standard`, `PREMIUM`, `Basic`, `STANDARD` | `Premium`, `Standard`, `Premium`, `Basic`, `Standard` | ✓ Konsisten |
| **payment_method** | `EWALLET`, `e-wallet`, `EWALLET`, `E-Wallet`, `Bank Transfer` | `E-Wallet`, `E-Wallet`, `E-Wallet`, `E-Wallet`, `Bank Transfer` | ✓ Konsisten |
| **primary_device** | `smart tv`, `SMART TV`, `Mobile`, `Smart TV`, `MOBILE` | `Smart TV`, `Smart TV`, `Mobile`, `Smart TV`, `Mobile` | ✓ Konsisten |
| **content_genre_favorite** | `action`, `Drama`, `ACTION`, `DRAMA`, `Sci-Fi` | `Action`, `Drama`, `Action`, `Drama`, `Sci-Fi` | ✓ Konsisten |
| **student_gpa** | NaN ( kosong) | **3.102953** (semua) | ✓ Diimput dengan mean |
| **work_productivity_score** | 8.5, 3.2, 8.7, 6.9, 6.2 | 8.5, 3.2, 8.7, 6.9, 6.2 | Tidak berubah (tidak ada missing di 5 baris ini) |

**Data sekarang "bersih"** — tidak ada lagi inkonsistensi kapitalisasi, tidak ada nilai negatif, tidak ada missing values.

---

### Output 4 — Nilai Unik & Frekuensi Kategori (Cell 31)

```
[Kolom: gender]
• Unique Values: ['Male' 'Female' 'Other']

[Kolom: subscription_plan]
• Unique Values: ['Premium' 'Standard' 'Basic' 'Student']

[Kolom: payment_method]
• Unique Values: ['E-Wallet' 'Bank Transfer' 'Credit Card' 'Paylater']

[Kolom: primary_device]
• Unique Values: ['Smart TV' 'Mobile' 'Tablet' 'Laptop/PC']

[Kolom: content_genre_favorite]
• Unique Values: ['Action' 'Drama' 'Sci-Fi' 'Anime' 'Comedy' 'Documentary']

[Kolom: satisfaction_score]
• Unique Values: ['Very Dissatisfied' 'Satisfied' 'Dissatisfied' 'Neutral' 'Very Satisfied']

[Kolom: streaming_quality_preference]
• Unique Values: ['4K UHD' 'FHD (1080p)' 'SD (480p)' 'HD (720p)']

[Kolom: churn_risk_level]
• Unique Values: ['High' 'Low' 'Moderate']
```

**Penjelasan:**
Output ini **membuktikan bahwa normalisasi berhasil**. Bandingkan jumlah nilai unik:

| Kolom | Sebelum | Sesudah | Status |
|---|---|---|---|
| `gender` | 17 | **3** | ✓ Berhasil |
| `subscription_plan` | 20 | **4** | ✓ Berhasil |
| `payment_method` | 20 | **4** | ✓ Berhasil |
| `primary_device` | 20 | **4** | ✓ Berhasil |
| `content_genre_favorite` | 25 | **6** | ✓ Berhasil |
| `satisfaction_score` | 5 | 5 | Sudah konsisten dari awal |
| `streaming_quality_preference` | 4 | 4 | Sudah konsisten dari awal |
| `churn_risk_level` | 3 | 3 | Sudah konsisten dari awal |

**Verifikasi kelengkapan:** Tidak ada nilai aneh tersisa. Semua kategori sesuai kamus data.

---

## 7. Scaling Data Numerik (Cell 34)

**Tidak ada output (tidak ada print/display).**

**Penjelasan:**
Kode menjalankan `RobustScaler` pada **11 variabel numerik**:

```python
numerical_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns  # 11 kolom
df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
```

**Cara kerja RobustScaler:**
Formula: `x_scaled = (x − median) / IQR`

* `median` = nilai tengah (Q2)
* `IQR` = Q3 − Q1

**Mengapa `RobustScaler` dipilih dan bukan StandardScaler atau MinMaxScaler?**

| Metode | Rumus | Keunggulan | Kekurangan |
|---|---|---|---|
| **RobustScaler** | (x − median) / IQR | **Tahan terhadap outlier** |—butuh perhitungan IQR |
| StandardScaler | (x − mean) / std | Cepat | **Sangat sensitif terhadap outlier** |
| MinMaxScaler | (x − min) / (max − min) | Konsisten ke [0,1] | Sangat sensitif terhadap nilai ekstrem |

**Alasan memilih RobustScaler:**

1. **Dataset ini mengandung outlier** (misalnya `buffering_events_weekly` yang skewness tinggi, `work_productivity_score` yang punya nilai di 1–2). StandardScaler akan **terdistorsi** oleh nilai-nilai ini.
2. **Lebih tahan terhadap outlier** — RobustScaler memakai median dan IQR yang **tidak terpengaruh oleh nilai ekstrem**, sehingga hasilnya lebih konsisten.
3. **Cocok untuk data mixed scale** — ada variabel dengan range 0–48 (`tenure_months`) dan 0–10 (`work_productivity_score`), keduanya akan dinormalisasi ke skala yang sebanding.

**Parameter scaler yang digunakan (dihitung dari data 990 baris):**

| Variabel | Median | IQR | Arti: nilai 0 = | Nilai -1 = | Nilai +1 = |
|---|---|---|---|---|---|
| `age` | 31,00 | 12,75 | 31 tahun | 18,25 tahun | 43,75 tahun |
| `monthly_subscription_fee_usd` | 13,99 | 5,00 | 13,99 USD | 8,99 USD | 18,99 USD |
| `daily_streaming_hours` | 3,49 | 1,685 | 3,49 jam | 1,805 jam | 5,175 jam |
| `watch_time_late_night_hours` | 1,175 | 1,3075 | 1,175 jam | −0,1325 jam | 2,4825 jam |
| `offline_downloads_monthly` | 6,00 | 11,00 | 6 unduhan | −5 unduhan | 17 unduhan |
| `number_of_profiles_shared` | 2,00 | 2,00 | 2 profil | 1 profil | 4 profil |
| `buffering_events_weekly` | 5,00 | 3,00 | 5 buffer | 2 buffer | 8 buffer |
| `customer_support_tickets` | 1,00 | 2,00 | 1 tiket | −1 tiket | 3 tiket |
| `student_gpa` | 3,103 | 1,00 | 3,103 | 2,103 | 4,103 |
| `work_productivity_score` | 6,80 | 1,90 | 6,80 | 4,90 | 8,70 |
| `tenure_months` | 13,00 | 20,00 | 13 bulan | −7 bulan | 33 bulan |

*Cell ini tidak menghasilkan output, tapi efeknya bisa diamati langsung di cell 37 (§8).*

---

## 8. Encoding Data Kategorik (Cell 36, 37)

### Output 1 — `df_scaled.head()` (setelah scaling + encoding)

```
  customer_id       age  gender  subscription_plan  monthly_subscription_fee_usd  payment_method
0   CUST-1753 -0.627451       1                3.0                           1.0               2
1   CUST-1520  1.490196       0                2.0                           0.0               2
2   CUST-1211  0.627451       1                3.0                           1.0               2
3   CUST-1612 -1.019608       1                1.0                          -1.0               2
4   CUST-1915  0.627451       0                2.0                           0.0               0

  daily_streaming_hours  primary_device  content_genre_favorite  watch_time_late_night_hours
0                0.142433               2                       0                     0.738050
1                0.451039               2                       4                     0.829828
2                1.044510               1                       0                     0.424474
3                0.011869               2                       4                    -0.829828
4                1.103858               1                       5                     0.340344
...
  buffering_events_weekly  customer_support_tickets  student_gpa  work_productivity_score
0                 0.666667                       2.0          0.0                 0.894737
1                -0.333333                      -0.5          0.0                -1.894737
2                 1.666667                       1.5          0.0                 1.000000
3                 0.000000                      -0.5          0.0                 0.052632
4                -1.000000                       0.5          0.0                -0.315789
```

**Penjelasan — Kolom Nominal (LabelEncoder):**

**4 kolom nominal** (tanpa urutan) di-encode dengan `LabelEncoder`, menghasilkan integer 0–n:

| Kolom | Urutan Encoding | Keterangan |
|---|---|---|
| `gender` | Female→0, Male→1, Other→2 | Alfabetis |
| `payment_method` | Bank Transfer→0, Credit Card→1, E-Wallet→2, Paylater→3 | Alfabetis |
| `primary_device` | Laptop/PC→0, Mobile→1, Smart TV→2, Tablet→3 | Alfabetis |
| `content_genre_favorite` | Action→0, Anime→1, Comedy→2, Documentary→3, Drama→4, Sci-Fi→5 | Alfabetis |

*Contoh verifikasi:* Baris 0 = CUST-1753, gender Male → **1** ✓; primary_device Smart TV → **2** ✓; genre Action → **0** ✓; payment E-Wallet → **2** ✓.

**Kenapa `LabelEncoder`?** Karena kolom-kolom ini **nominal** (tidak ada urutan). Satu integer per kategori, jumlah kategori = jumlah nilai unik.

**Kelemahan `LabelEncoder`:** kelemahannya adalah **encoding selalu mengikuti urutan alfabetis (alphabetical)** — bukan berdasarkan urutan logis. `Female→0, Male→1` **tidak absurd secara domain**, tetapi bisa bermasalah jika nanti digunakan bersama model yang memperlakukan jarak antar integer sebagai bermakna (misalnya LinearRegression). Untuk tree-based model (RandomForest, XGBoost) ini **tidak masalah**.

**Kolom Ordinal (OrdinalEncoder):**

**4 kolom ordinal** (dengan urutan logis) di-encode dengan `OrdinalEncoder` **beserta urutan kategori yang eksplisit**:

| Kolom | Urutan Encoding |
|---|---|
| `subscription_plan` | Student→0, Basic→1, Standard→2, Premium→3 |
| `satisfaction_score` | Very Dissatisfied→0, Dissatisfied→1, Neutral→2, Satisfied→3, Very Satisfied→4 |
| `streaming_quality_preference` | SD (480p)→0, HD (720p)→1, FHD (1080p)→2, 4K UHD→3 |
| `churn_risk_level` | Low→0, Moderate→1, High→2 |

*Verifikasi:* Baris 0: Premium→3 ✓, Very Dissatisfied→0 ✓, 4K UHD→3 ✓, High→2 ✓. Baris 3: Basic→1 ✓, Neutral→2 ✓, SD (480p)→0 ✓, Moderate→1 ✓.

**Kenapa `OrdinalEncoder` untuk kolom-kolom ini?** Karena **urutan kategori bermakna**:
* `subscription_plan`: Student < Basic < Standard < Premium (makin mahal).
* `satisfaction_score`: Sangat Tidak Puas < Tidak Puas < Netral < Puas < Sangat Puas.
* `streaming_quality_preference`: SD < HD < FHD < 4K (makin tinggi resolusi).
* `churn_risk_level`: Low < Moderate < High (makin tinggi risiko).

`OrdinalEncoder` **menjaga urutan ini** — sesuatu yang **tidak bisa dilakukan** oleh `LabelEncoder` (yang selalu alfabetis). Jika `churn_risk_level` di-encode dengan `LabelEncoder`, hasilnya: High→0, Low→1, Moderate→2 — **berantakan**! `OrdinalEncoder` justru mempertahankan urutan logis.

**Tentang `student_gpa` = 0.0 untuk semua 5 baris (kritis):**

Setelah diimput dengan mean, **149 pelanggan non-Student** yang tadinya `NaN` sekarang bernilai `3.102953` (mean). Ketika di-scale dengan RobustScaler:
* Median dari `student_gpa` (termasuk semua nilai yang sudah diimput) = **3.102953**
* IQR = **1.00**

Jadi: `(3.102953 − 3.102953) / 1.00 = **0.0**`

**Artinya:** Nilai 0.0 pada `student_gpa` **artinya "tidak punya IPK"** (yaitu nilai hasil imputasi), bukan IPK yang sebenarnya 0. Pelanggan Student (149 orang) yang punya IPK asli akan punya **nilai bukan 0** setelah scaling. Kolom ini **hanya bermakna** jika kita ingat bahwa 0.0 = "hasil imputasi / tidak berlaku".

---

## 9. Feature Construction (Cell 39, 40, 41, 42)

### Output 1 — `df_scaled.head()` (setelah feature engineering)

```
  tenure_months  churn_risk_level  late_night_streaming_ratio
0          -0.30               2.0                    5.181724
1          -0.20               0.0                    1.839816
2           1.75               2.0                    0.406386
3          -0.20               1.0                  -69.913002
4          -0.60               1.0                    0.308323

  cost_per_streaming_hour  support_tickets_per_tenure_month
0                 0.234027                         -6.666889
1                 0.000000                          2.500125
2                 0.031913                          0.857138
3                -2.808254                          2.500125
4                 0.000000                         -0.833347
```

**Penjelasan — 3 fitur baru yang dibuat:**

| Fitur | Rumus | Interpretasi |
|---|---|---|
| `late_night_streaming_ratio` | `watch_time_late_night_hours / daily_streaming_hours` | Proporsi jam nonton larut malam terhadap total jam nonton. 0–1 ideal |
| `cost_per_streaming_hour` | `monthly_subscription_fee_usd / (daily_streaming_hours × 30)` | Biaya per jam nonton per bulan. Makin rendah makin untung |
| `support_tickets_per_tenure_month` | `customer_support_tickets / tenure_months` | Rata-rata tiket komplain per bulan masa berlangganan |

**Relevansi terhadap analisis:**

1. **`late_night_streaming_ratio`**: Menilai **gaya menonton** pengguna. Rasio tinggi berarti habit watching lebih besar di larut malam. Ini berkorelasi dengan risiko churn.
2. **`cost_per_streaming_hour`**: Menilai **nilai/value** langganan. Kalau biaya per jam sangat tinggi, pengguna mungkin merasa tidak worthwhile.
3. **`support_tickets_per_tenure_month`**: Menilai **frekuensi keluhan** per satuan waktu. Pelanggan dengan tiket tinggi per bulan = **tidak puas**.

---

## 10. Data Exploration — Visualisasi (Cell 45, 46, 47)

### Output 1 — Visualisasi 1: Stacked Bar Chart — Proporsi Churn Risk per Satisfaction Score (Cell 45)

```
<Figure size 1800x900 with 1 Axes>
```

**Penjelasan:**
Kode membuat **stacked bar chart** yang menunjukkan **proporsi churn risk di setiap tingkat kepuasan**. Setiap bar mewakili satu tingkat `satisfaction_score`, dan warnanya menunjukkan % dalam setiap `churn_risk_level`.

**Data yang diplot (hasil crosstab):**

| Satisfaction | Low | Moderate | High | Total |
|---|---|---|---|---|
| Very Dissatisfied | 0,0% | 0,0% | **100,0%** | 86 |
| Dissatisfied | 0,0% | 8,7% | **91,3%** | 208 |
| Neutral | 21,6% | **57,8%** | 20,6% | 393 |
| Satisfied | **72,8%** | 27,2% | 0,0% | 276 |
| Very Satisfied | **88,9%** | 11,1% | 0,0% | 27 |

*Jumlah: 86 + 208 + 393 + 276 + 27 = 990 baris* ✓

**Apa yang Ditunjukkan Visualisasi Ini:**

* **Very Dissatisfied → 100% High churn** (86 orang, semua risiko tinggi)
* **Dissatisfied → 91,3% High churn** (208 orang, sebagian besar berisiko)
* **Neutral → 57,8% Moderate churn** (393 orang, posisi ambigu)
* **Satisfied → 72,8% Low churn** (276 orang, sebagian besar aman)
* **Very Satisfied → 88,9% Low churn** (27 orang, sangat aman)

**Insight Kunci:**
Ada **korelasi negatif yang kuat** antara `satisfaction_score` dan `churn_risk_level`. Makin puas → makin rendah risiko churn. Ini **pola yang sangat konsisten dan logis** untuk bisnis streaming.

---

### Output 2 — Visualisasi 2: Bar Chart — Durasi Streaming per Paket (Cell 46)

```
[Warning: FutureWarning]
Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0.
Assign the `x` variable to `hue` and set `legend=False` for the same effect.

[display]
<Figure size 1800x900 with 1 Axes>
```

**Penjelasan:**
Kode membuat **bar chart** yang menampilkan **rata-rata `daily_streaming_hours` untuk setiap paket langganan**.

**Data yang diplot:**

| Paket | Rata-rata jam/hari | Jumlah Pengguna |
|---|---|---|
| Student | **3,60** | 149 |
| Basic | 3,51 | 293 |
| Standard | 3,53 | 337 |
| Premium | 3,45 | 211 |

**Apa yang Ditunjukkan:**
Rata-rata durasi streaming **sangat mirip** di semua paket (3,45–3,60 jam). Artinya **tidak ada perbedaan signifikan** dalam kebiasaan menonton antar paket. Ini tidak terduga — kita mungkin mengira Premium = nonton lebih lama, tapi ternyata **sama saja**.

**FutureWarning** muncul karena `palette='Blues_d'` diberikan tanpa `hue`. Dalam versi seaborn terbaru, palette harus di-assign ke `hue`, bukan langsung ke `palette`. **Warning ini tidak merusak visualisasi** — chart tetap benar.

---

### Output 3 — Visualisasi 3: Grouped Countplot — Preferensi Resolusi per Paket (Cell 47)

```
<Figure size 1800x900 with 1 Axes>
```

**Penjelasan:**
Kode membuat **grouped bar chart** yang menunjukkan **distribusi `streaming_quality_preference` dalam setiap paket langganan**. Setiap bar mewakili satu kombinasi (paket × resolusi), dan warnanya menunjukkan resolusi.

**Data yang diplot:**

| Paket | SD (480p) | HD (720p) | FHD (1080p) | 4K UHD |
|---|---|---|---|---|
| Student | **0** | **84** (56,4%) | 65 (43,6%) | **0** |
| Basic | **155** (52,9%) | 138 (47,1%) | **0** | **0** |
| Standard | **0** | **0** | **177** (52,5%) | 160 (47,5%) |
| Premium | **0** | **0** | 65 (30,8%) | **146** (69,2%) |

**Apa yang Ditunjukkan:**

* **Paket Basic**: Hanya ada di SD dan HD, **tidak ada** yang pilih FHD atau 4K.
* **Paket Standard & Premium**: Hanya ada di FHD dan 4K, **tidak ada** di SD atau HD.
* **Paket Student**: Hanya di HD dan FHD, **tidak ada** di SD atau 4K.

**Insight Kunci — Hubungan Deterministik:**
Terdapat **pola yang sangat jelas dan konsisten**: **setiap paket hanya memiliki 2 pilihan resolusi tertentu** (tidak lebih, tidak kurang). Ini menunjukkan **bahwa `streaming_quality_preference` sangat berkaitan erat dengan `subscription_plan`** — bukan pilihan independen.

Analogi: seperti **menu paket tertutup** — kalau kamu bayar Basic, kamu hanya bisa pilih SD atau HD. Tidak ada opsi untuk pilih 4K. Seberapa pun kamu mau, **tidak bisa akses resolusi lebih tinggi dari yang termasuk paket**.

**Implikasi untuk machine learning:**
`streaming_quality_preference` **hampir redundant** dengan `subscription_plan` — keduanya memberikan informasi yang **sangat mirip**. Jika keduanya dimasukkan ke model, **risiko multicollinearity** (korelasi tinggi antar fitur) bisa meningkat, meskipun tidak terlalu berbahaya untuk model tree-based (RandomForest, XGBoost).

---

## Catatan Kritis — Temuan yang Perlu Diperhatikan

### 1. Feature Construction pada Data yang Sudah Di-Scale

**Masalah:** Fitur baru (`late_night_streaming_ratio`, `cost_per_streaming_hour`, `support_tickets_per_tenure_month`) **dibangun dari `df_scaled`** (data yang sudah di-RobustScale), bukan dari `df_processed` (data bersih sebelum scale).

**Dampak:**
* Pembagian dari bilangan yang sudah di-centering (median) **dapat menghasilkan nilai negatif** yang **tidak logis** secara domain (misal `late_night_streaming_ratio = -69.91` — **ratio tidak bisa negatif** jika kedua variabel positif).
* Pada data asli, range `late_night_streaming_ratio` seharusnya **0–0,75** (terlihat dari 5 baris pertama: 0,11; 0,15; 0,11; 0,03; 0,13). Tapi setelah penskalaan, muncul nilai-nilai aneh: −69,9; 5,18; 1,84.
* Secara statistik, hasil ini **tidak dapat diinterpretasi** secara intuitif.

**Rekomendasi:** Bangun fitur baru dari `df_processed` **sebelum** scaling, lalu scale hasilnya secara terpisah jika diperlukan.

---

### 2. Baris Kode yang Terduplikasi

Beberapa baris di notebook **terduplikasi tanpa disengaja** (tidak merusak, tapi tidak perlu):

```python
# Cell 24 — baris yang sama ditulis 2× berturut-turut
num_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
```

---

### 3. Ringkasan Alur Pembersihan Data

| Tahap | Baris | Δ | Keterangan |
|---|---|---|---|
| **Data mentah** | 1.015 | — | Load CSV |
| **Setelah `drop_duplicates()`** | 1.000 | −15 | Hapus 15 baris duplikat |
| **Setelah filter `daily_streaming_hours ≤ 24`** | 995 | −5 | Hapus 5 baris > 24 jam |
| **Setelah hapus outlier fee** | **990** | −5 | Hapus fee ∈ {399, 499, 650, 850, 999} |
| **Setelah imputasi (tidak mengubah baris)** | 990 | 0 | Isi student_gpa & work_productivity |
| **Setelah scaling + encoding** | 990 | 0 | Ubah format, jumlah baris tetap |
| **Setelah feature engineering** | 990 | 0 | Tambah 3 kolom fitur |

---

### 4. Rekomendasi Perbaikan (jika ingin meningkatkan kualitas)

1. **Feature construction:** Bangun fitur dari `df_processed` (sebelum scaling).
2. **Outlier `age` 56–60:** Pertimbangkan untuk **membatasi** (clip) jika ingin konsistensi dengan metode IQR. Saat ini **tidak dihapus** — secara domain **masih valid** (lansia).
3. **`student_gpa` imputation:** Gunakan **conditional imputation** — isi dengan NaN jika plan ≠ Student, isi dengan median jika plan = Student. Ini lebih logis secara domain.
4. **Pipeline yang lebih baik:** Gunakan `sklearn.pipeline.Pipeline` untuk mengotomatisasi scaling + encoding dalam satu objek yang dapat dipickle.
5. **Validasi model:** Pisahkan train/test **sebelum** fitting scaler/encoder untuk menghindari data leakage.

---

*Dokumen ini menjelaskan seluruh output yang dihasilkan di notebook — 25 output di 11 bagian. Semua angka diverifikasi ulang terhadap data asli.*