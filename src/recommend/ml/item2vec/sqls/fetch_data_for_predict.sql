SELECT
     w1.uid user_id
    ,w1.datetime
    ,w1.edinet_code
FROM
    company_view_log w1
WHERE
    uid = :user_id