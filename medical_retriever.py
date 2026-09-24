from data_loader import medical_data


def retrieve_medical_information(
    entities
):

    results = []


    for entity in entities:

        text = entity["text"]

        category = entity["category"]


        # Find entity in knowledge base

        if category not in medical_data:

            continue


        if text not in medical_data[
            category
        ]:

            continue


        information = medical_data[
            category
        ][text]


        result = {

            "entity": text,

            "type": information[
                "type"
            ],

            "description": information[
                "description"
            ]

        }


        # Include related information
        # when available

        if "related" in information:

            result["related"] = (
                information["related"]
            )


        results.append(
            result
        )


    return results