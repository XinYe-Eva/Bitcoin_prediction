from tardis_dev import datasets

if __name__ == "__main__":
    api_key = (
        "TD.DVnuaoaxFNRT1xnb.-c5pe6luxohV8Qa.RxOZfHDkhJrG955"
        ".c8KctNpdMxM8OAK.4wvtFAQD745RNQc.puci"
    )
    datasets.download(
        exchange="binance",
        data_types=["book_snapshot_5"],
        from_date="2022-10-30",
        to_date="2022-10-31",
        symbols=["BTCUSDT"],
        api_key=api_key,
    )
