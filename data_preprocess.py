import pandas as pd

from src.preprocess_utils import preprocess_to_weekly_sales

if __name__ == "__main__":
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

    for prod_id in product_id_list:
        product_orders = meta_order[meta_order["PRODUCT_ID"] == prod_id]
        product_weekly_sales = preprocess_to_weekly_sales(product_orders, "2022-01-01", "2022-04-30")

        if all_sales is not None:
            all_sales = pd.concat([all_sales, product_orders], ignore_index=True)
        else:
            all_sales = product_orders

    all_sales = all_sales.melt(
        id_vars=["PRODUCT_ID"], var_name="week", value_name="sales"
    )

    all_sales.rename(columns={"PRODUCT_ID": "product"}, inplace=True)
    all_sales.to_csv("sales_data.csv", index=False)


# start_date = product_orders["CHECKOUT_DATE"].min()
#     start_date = datetime.strptime(start_date, "%Y-%m-%d")
#     start_date = str(start_date.replace(day=1).date())

#     end_month = product_orders["CHECKOUT_DATE"].max().month

#     start_date = f""
