WITH filer_name_table AS (
    /* 
    企業名と edinet_code の対応テーブルを作成する 
    
    各企業で edinet_code は不変だが企業名は変更される可能性があるため、
    最新の有価証券報告書で指定された企業名を採用する。
    */
    SELECT
         w1.edinet_code
        ,w1.filer_name
    FROM
        doc_info w1
    INNER JOIN (
        SELECT
            w2.edinet_code
            ,MAX(w2.doc_end_date_of_fiscal_year) AS latest_date
        FROM
            doc_info w2
        GROUP BY
            w2.edinet_code
    ) w2
        ON 
            w1.edinet_code = w2.edinet_code
            AND w1.doc_end_date_of_fiscal_year = w2.latest_date

)

SELECT
    /* 企業の閲覧履歴情報を取得する */
     w1.uid user_id
    ,w1.datetime
    ,w1.edinet_code
    ,w2.filer_name
FROM
    company_view_log w1
INNER JOIN
    filer_name_table w2
    ON w1.edinet_code = w2.edinet_code
WHERE
    w1.datetime >= DATE_SUB(NOW(), INTERVAL 3 MONTH)