"""Ekstrak komentar # dari file Python ke markdown terpisah, lalu hapus dari kode.
Docstring (""triple-quoted"") TIDAK disentuh -- itu bukan komentar, itu dokumentasi.

Pakai tokenize (bukan regex) supaya aman -- tidak merusak string yang
kebetulan mengandung karakter '#'.
"""
import argparse
import io
import tokenize
from pathlib import Path


def proses(path_masuk: Path, path_keluar_kode: Path, path_keluar_catatan: Path) -> None:
    kode = path_masuk.read_text(encoding="utf-8")
    tokens = list(tokenize.generate_tokens(io.StringIO(kode).readline))

    catatan = []
    baris = kode.splitlines(keepends=True)
    # Tandai posisi karakter yang perlu dihapus (komentar saja), baris tidak dihapus.
    hapus_rentang = []  # (baris_index0, kolom_mulai, kolom_akhir)
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            baris_no, kolom_mulai = tok.start
            _, kolom_akhir = tok.end
            teks_komentar = tok.string
            catatan.append(f"- Baris {baris_no}: `{teks_komentar}`")
            hapus_rentang.append((baris_no - 1, kolom_mulai, kolom_akhir))

    for idx, kolom_mulai, kolom_akhir in hapus_rentang:
        baris_asli = baris[idx]
        baris_baru = baris_asli[:kolom_mulai] + baris_asli[kolom_akhir:]
        # Kalau setelah dihapus cuma spasi kosong + newline, dan baris awalnya
        # cuma komentar (tidak ada kode lain), kosongkan total barisnya nanti.
        baris[idx] = baris_baru

    kode_bersih = "".join(baris)
    # Rapikan baris yang isinya cuma whitespace sisa dari komentar full-line.
    kode_bersih_lines = []
    for line in kode_bersih.splitlines(keepends=True):
        if line.strip() == "" and line != "\n" and line.strip("\n") == "":
            kode_bersih_lines.append("\n" if line.endswith("\n") else "")
        else:
            kode_bersih_lines.append(line)
    kode_bersih = "".join(kode_bersih_lines)

    path_keluar_kode.write_text(kode_bersih, encoding="utf-8")

    isi_catatan = f"# Catatan/komentar asli dari `{path_masuk.name}`\n\n" + "\n".join(catatan)
    path_keluar_catatan.write_text(isi_catatan, encoding="utf-8")

    print(f"{path_masuk.name}: {len(catatan)} komentar dipindah ke {path_keluar_catatan.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="File .py yang mau dibersihkan")
    args = parser.parse_args()

    for nama in args.files:
        path_masuk = Path(nama)
        path_keluar_kode = path_masuk.with_name(path_masuk.stem + "_bersih.py")
        path_keluar_catatan = path_masuk.with_name(path_masuk.stem + "_catatan.md")
        proses(path_masuk, path_keluar_kode, path_keluar_catatan)


if __name__ == "__main__":
    main()
