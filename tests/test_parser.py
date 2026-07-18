from src.resume_parser.parser_service import ParserService

print("Starting parser...")

parser = ParserService()

print("Parser created.")

try:
    doc = parser.parse_file("tests/sample_resume.pdf")

    print("parse_file returned:")
    print(doc)
    print(type(doc))

    if doc is not None:
        print(doc.model_dump_json(indent=2))
    else:
        print("parse_file returned None")

except Exception as e:
    import traceback
    traceback.print_exc()