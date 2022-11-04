import pandas as pd

if __name__ == '__main__':
    meta_order = pd.read_csv("data/data_order.csv")
    meta_order["CHECKOUT_DATE"] = pd.to_datetime(
        meta_order["CHECKOUT_DATE"], format="%Y-%m-%d"
    )

    # drop irrelevant keys
    meta_order = meta_order.drop(
        [
            "VARIANT_CASE_PRICE_CENTS",
            "REGION_ID",
            "BRAND_ID",
            "ORDER_ID",
            "STORE_ID",
            "PRODUCT_VARIANT_ID",
        ],
        axis=1,
    )

    product_id_list = meta_order["PRODUCT_ID"].unique()
    all_sales = None

    for prod_id in product_id_list[:5]:

        ## ============================
        ## Fill out missing days
        ## ============================
        product_orders = meta_order.loc[meta_order["PRODUCT_ID"] == prod_id].sort_values(
            by="CHECKOUT_DATE"
        )
        product_orders = (
            product_orders.groupby(["CHECKOUT_DATE", "PRODUCT_ID"])
            .sum()
            .sort_values(by="CHECKOUT_DATE")
            .reset_index()
        )

        r = pd.date_range(start="2022-01-01", end="2022-04-30", freq="D")
        fill_values = {"PRODUCT_ID": prod_id, "QUANTITY": 0}
        product_orders = (
            product_orders.set_index("CHECKOUT_DATE")
            .reindex(r)
            .fillna(value=fill_values)
            .rename_axis("CHECKOUT_DATE")
            .reset_index()
        )

        ## ============================
        ## Convert daily to weekly data
        ## ============================
        product_orders = (
            product_orders.groupby(
                [pd.Grouper(key="CHECKOUT_DATE", freq="W-MON"), "PRODUCT_ID"]
            )
            .sum()
            .sort_values(by="CHECKOUT_DATE")
        )
        product_orders = product_orders.reset_index().drop(
            ["CHECKOUT_DATE", "PRODUCT_ID"], axis=1
        )
        product_orders = product_orders[
            :-1
        ]  # Remove the last week because data for the last days are missing

        ## ============================
        ## Concat
        ## ============================
        product_orders = product_orders.T.reset_index().drop(["index"], axis=1)
        product_orders.insert(loc=0, column="PRODUCT_ID", value=[prod_id])

        if all_sales is not None:
            all_sales = pd.concat([all_sales, product_orders], ignore_index=True)
        else:
            all_sales = product_orders

    all_sales = all_sales.melt(id_vars=["PRODUCT_ID"], var_name="WEEK", value_name="SALES")

    train_weeks = round(all_sales.max()['WEEK'] * 0.8)
    train = all_sales.loc[all_sales["WEEK"] < train_weeks]
    val = all_sales.loc[all_sales["WEEK"] >= train_weeks]

    print(all_sales)
