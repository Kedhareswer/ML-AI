"""Record actual browser interactions; requires playwright and its video support."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--chromium",default="/usr/bin/chromium")
    parser.add_argument("--url",default="http://127.0.0.1:7860")
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    evidence=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.chromium,headless=True,args=["--no-sandbox","--disable-dev-shm-usage"])
        context=browser.new_context(viewport={"width":1200,"height":1100},record_video_dir=str(output/"video"),record_video_size={"width":1200,"height":1100})
        page=context.new_page()
        errors=[]
        page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("document.querySelector('#badge').textContent.includes('MB')")
        for name,mode,rows in [
            ("neural-4x4","model",["S...",".X..","..X.","...G"]),
            ("neural-detour","model",["S...","XXX.","....","GXXX"]),
            ("guided-8x8","guided",["S.......","XXXXXXX.","........",".XXXXXXX","........","XXXXXXX.","........","GXXXXXXX"]),
        ]:
            page.locator('#mode').select_option(mode)
            page.locator('#maze').fill('\n'.join(rows))
            with page.expect_response('**/api/solve') as pending:
                page.locator('#solve').click()
            result=pending.value.json()
            assert result['solved'],(name,result)
            page.wait_for_function("document.querySelector('#status').textContent.includes('Verified solution')")
            page.wait_for_timeout(len(result['history'][-1]['path'])*160+500)
            page.screenshot(path=str(output/f'{name}.png'),full_page=True)
            evidence.append({'name':name,'mode':mode,'rows':rows,'response':result})
        assert not errors,errors
        video=page.video
        context.close()
        video.save_as(str(output/'solving-demo.webm'))
        video.delete()
        browser.close()
    (output/'browser-evidence.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'screenshots':3,'recording':str(output/'solving-demo.webm'),'javascript_errors':errors}))

if __name__=='__main__':
    main()
