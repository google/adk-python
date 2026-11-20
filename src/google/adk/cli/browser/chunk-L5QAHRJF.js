import{a as nt}from"./chunk-CDENLQJG.js";import{a as lt}from"./chunk-OA7KAXCR.js";import"./chunk-YSDWRGGV.js";import"./chunk-S6ALDCTQ.js";import"./chunk-SUCP2KH6.js";import"./chunk-KD6NNSEH.js";import"./chunk-WZ6ENJVG.js";import"./chunk-QBCO4P2K.js";import"./chunk-QNYO23UL.js";import"./chunk-QCIEL2KM.js";import"./chunk-WMENXGN7.js";import"./chunk-LHC2GLSA.js";import"./chunk-UQKFLMSW.js";import"./chunk-H7M4KRZ2.js";import"./chunk-L32LDBVN.js";import"./chunk-VZ4YUAP7.js";import"./chunk-LCQFZR6Q.js";import"./chunk-R5XN4DWF.js";import{a as rt}from"./chunk-KEJFSQGZ.js";import{n as it,o as ot}from"./chunk-3DH3RDF5.js";import"./chunk-RTKEQAY7.js";import{O as X,T as Z,U as j,V as q,W as J,X as K,Y as Q,Z as Y,_ as tt,k as V}from"./chunk-SQTYPNHV.js";import{G as R,J as at,b as T,n as et}from"./chunk-Z7SBFKWA.js";import{a as l}from"./chunk-YUSHYV7C.js";import{g as U}from"./chunk-NCEQPEJE.js";var st=V.pie,L={sections:new Map,showData:!1,config:st},b=L.sections,O=L.showData,xt=structuredClone(st),wt=l(()=>structuredClone(xt),"getConfig"),Ct=l(()=>{b=new Map,O=L.showData,Z()},"clear"),$t=l(({label:t,value:a})=>{if(a<0)throw new Error(`"${t}" has invalid value: ${a}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);b.has(t)||(b.set(t,a),T.debug(`added new section: ${t}, with value: ${a}`))},"addSection"),Dt=l(()=>b,"getSections"),yt=l(t=>{O=t},"setShowData"),Tt=l(()=>O,"getShowData"),ct={getConfig:wt,clear:Ct,setDiagramTitle:Q,getDiagramTitle:Y,setAccTitle:j,getAccTitle:q,setAccDescription:J,getAccDescription:K,addSection:$t,getSections:Dt,setShowData:yt,getShowData:Tt},bt=l((t,a)=>{nt(t,a),a.setShowData(t.showData),t.sections.map(a.addSection)},"populateDb"),At={parse:l(t=>U(null,null,function*(){let a=yield lt("pie",t);T.debug(a),bt(a,ct)}),"parse")},kt=l(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieCircle.highlighted{
    scale: 1.05;
    opacity: 1;
  }
  .pieCircle.highlightedOnHover:hover{
    transition-duration: 250ms;
    scale: 1.05;
    opacity: 1;
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),_t=kt,zt=l(t=>{let a=[...t.values()].reduce((n,m)=>n+m,0),W=[...t.entries()].map(([n,m])=>({label:n,value:m})).filter(n=>n.value/a*100>=1);return at().value(n=>n.value).sort(null)(W)},"createPieArcs"),Et=l((t,a,W,F)=>{T.debug(`rendering pie chart
`+t);let n=F.db,m=tt(),h=ot(n.getConfig(),m.pie),H=40,i=18,c=4,S=450,x=S,A=rt(a),$=A.append("g");$.attr("transform","translate("+x/2+","+S/2+")");let{themeVariables:o}=m,[M]=it(o.pieOuterStrokeWidth);M??=2;let dt=h.legendPosition,P=h.textPosition,gt=h.donutHole>0&&h.donutHole<=.9?h.donutHole:0,f=Math.min(x,S)/2-H,pt=R().innerRadius(gt*f).outerRadius(f),ht=R().innerRadius(f*P).outerRadius(f*P),w=$.append("g");w.append("circle").attr("cx",0).attr("cy",0).attr("r",f+M/2).attr("class","pieOuterCircle");let D=n.getSections(),ft=zt(D),ut=[o.pie1,o.pie2,o.pie3,o.pie4,o.pie5,o.pie6,o.pie7,o.pie8,o.pie9,o.pie10,o.pie11,o.pie12],k=0;D.forEach(e=>{k+=e});let G=ft.filter(e=>(e.data.value/k*100).toFixed(0)!=="0"),_=et(ut).domain([...D.keys()]);w.selectAll("mySlices").data(G).enter().append("path").attr("d",pt).attr("fill",e=>_(e.data.label)).attr("class",e=>{let r="pieCircle";return h.highlightSlice==="hover"?r+=" highlightedOnHover":h.highlightSlice===e.data.label&&(r+=" highlighted"),r}),w.selectAll("mySlices").data(G).enter().append("text").text(e=>(e.data.value/k*100).toFixed(0)+"%").attr("transform",e=>"translate("+ht.centroid(e)+")").style("text-anchor","middle").attr("class","slice");let mt=$.append("text").text(n.getDiagramTitle()).attr("x",0).attr("y",-(S-50)/2).attr("class","pieTitleText"),C=[...D.entries()].map(([e,r])=>({label:e,value:r})),u=$.selectAll(".legend").data(C).enter().append("g").attr("class","legend");u.append("rect").attr("width",i).attr("height",i).style("fill",e=>_(e.label)).style("stroke",e=>_(e.label)),u.append("text").attr("x",i+c).attr("y",i-c).text(e=>n.getShowData()?`${e.label} [${e.value}]`:e.label);let v=Math.max(...u.selectAll("text").nodes().map(e=>e?.getBoundingClientRect().width??0)),y=S,z=x+H,s=i+c,E=C.length*s;switch(dt){case"center":u.attr("transform",(e,r)=>{let d=s*C.length/2,g=-v/2-(i+c),p=r*s-d;return"translate("+g+","+p+")"});break;case"top":y+=E,u.attr("transform",(e,r)=>{let d=f,g=-v/2-(i+c),p=r*s-d;return`translate(${g}, ${p})`}),w.attr("transform",()=>`translate(0, ${E+s})`);break;case"bottom":y+=E,u.attr("transform",(e,r)=>{let d=-f-s,g=-v/2-(i+c),p=r*s-d;return"translate("+g+","+p+")"});break;case"left":z+=i+c+v,u.attr("transform",(e,r)=>{let d=s*C.length/2,g=-f-(i+c),p=r*s-d;return"translate("+g+","+p+")"}),w.attr("transform",()=>`translate(${v+i+c}, 0)`);break;default:z+=i+c+v,u.attr("transform",(e,r)=>{let d=s*C.length/2,g=12*i,p=r*s-d;return"translate("+g+","+p+")"});break}let B=mt.node()?.getBoundingClientRect().width??0,vt=x/2-B/2,St=x/2+B/2,N=Math.min(0,vt),I=Math.max(z,St)-N;A.attr("viewBox",`${N} 0 ${I} ${y}`),X(A,y,I,h.useMaxWidth)},"draw"),Rt={draw:Et},Nt={parser:At,db:ct,renderer:Rt,styles:_t};export{Nt as diagram};
