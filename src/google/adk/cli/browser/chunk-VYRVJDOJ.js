import{a as Q}from"./chunk-DMWOYWYQ.js";import{a as Y}from"./chunk-SRCUB3EX.js";import"./chunk-X43UMWSZ.js";import"./chunk-DZQRYCWR.js";import"./chunk-QXIJPCUK.js";import"./chunk-GO6ZTN3G.js";import"./chunk-7BGAJP6A.js";import"./chunk-PNXCYZAZ.js";import"./chunk-BWRTTESZ.js";import"./chunk-5JNRF4JL.js";import"./chunk-PKSFXE6N.js";import"./chunk-7ZGKZ6HH.js";import{a as H}from"./chunk-PDRDFWTH.js";import{B as J,C as K}from"./chunk-UKZIEWH5.js";import"./chunk-GP6TCC26.js";import{N as O,R as B,S as P,T as I,U as N,V as U,W as V,X,Y as Z,r as L}from"./chunk-37QI3DOO.js";import{K as C,N as q,g as o,i as h,r as j}from"./chunk-JRNAXTJ7.js";import"./chunk-F57K64GP.js";import"./chunk-URMDZFG4.js";import{j as G}from"./chunk-RMXJBC7V.js";var ee=L.pie,D={sections:new Map,showData:!1,config:ee},u=D.sections,y=D.showData,he=structuredClone(ee),ue=o(()=>structuredClone(he),"getConfig"),me=o(()=>{u=new Map,y=D.showData,B()},"clear"),ve=o(({label:e,value:a})=>{if(a<0)throw new Error(`"${e}" has invalid value: ${a}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);u.has(e)||(u.set(e,a),h.debug(`added new section: ${e}, with value: ${a}`))},"addSection"),xe=o(()=>u,"getSections"),Se=o(e=>{y=e},"setShowData"),we=o(()=>y,"getShowData"),te={getConfig:ue,clear:me,setDiagramTitle:V,getDiagramTitle:X,setAccTitle:P,getAccTitle:I,setAccDescription:N,getAccDescription:U,addSection:ve,getSections:xe,setShowData:Se,getShowData:we},Ce=o((e,a)=>{Q(e,a),a.setShowData(e.showData),e.sections.map(a.addSection)},"populateDb"),De={parse:o(e=>G(null,null,function*(){let a=yield Y("pie",e);h.debug(a),Ce(a,te)}),"parse")},ye=o(e=>`
  .pieCircle{
    stroke: ${e.pieStrokeColor};
    stroke-width : ${e.pieStrokeWidth};
    opacity : ${e.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${e.pieOuterStrokeColor};
    stroke-width: ${e.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${e.pieTitleTextSize};
    fill: ${e.pieTitleTextColor};
    font-family: ${e.fontFamily};
  }
  .slice {
    font-family: ${e.fontFamily};
    fill: ${e.pieSectionTextColor};
    font-size:${e.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${e.pieLegendTextColor};
    font-family: ${e.fontFamily};
    font-size: ${e.pieLegendTextSize};
  }
`,"getStyles"),$e=ye,Te=o(e=>{let a=[...e.values()].reduce((r,l)=>r+l,0),$=[...e.entries()].map(([r,l])=>({label:r,value:l})).filter(r=>r.value/a*100>=1);return q().value(r=>r.value).sort(null)($)},"createPieArcs"),Ae=o((e,a,$,T)=>{h.debug(`rendering pie chart
`+e);let r=T.db,l=Z(),A=K(r.getConfig(),l.pie),b=40,n=18,p=4,s=450,d=s,m=H(a),c=m.append("g");c.attr("transform","translate("+d/2+","+s/2+")");let{themeVariables:i}=l,[E]=J(i.pieOuterStrokeWidth);E??=2;let _=A.textPosition,g=Math.min(d,s)/2-b,ae=C().innerRadius(0).outerRadius(g),ie=C().innerRadius(g*_).outerRadius(g*_);c.append("circle").attr("cx",0).attr("cy",0).attr("r",g+E/2).attr("class","pieOuterCircle");let f=r.getSections(),re=Te(f),oe=[i.pie1,i.pie2,i.pie3,i.pie4,i.pie5,i.pie6,i.pie7,i.pie8,i.pie9,i.pie10,i.pie11,i.pie12],v=0;f.forEach(t=>{v+=t});let k=re.filter(t=>(t.data.value/v*100).toFixed(0)!=="0"),x=j(oe).domain([...f.keys()]);c.selectAll("mySlices").data(k).enter().append("path").attr("d",ae).attr("fill",t=>x(t.data.label)).attr("class","pieCircle"),c.selectAll("mySlices").data(k).enter().append("text").text(t=>(t.data.value/v*100).toFixed(0)+"%").attr("transform",t=>"translate("+ie.centroid(t)+")").style("text-anchor","middle").attr("class","slice");let ne=c.append("text").text(r.getDiagramTitle()).attr("x",0).attr("y",-(s-50)/2).attr("class","pieTitleText"),R=[...f.entries()].map(([t,w])=>({label:t,value:w})),S=c.selectAll(".legend").data(R).enter().append("g").attr("class","legend").attr("transform",(t,w)=>{let M=n+p,pe=M*R.length/2,ge=12*n,fe=w*M-pe;return"translate("+ge+","+fe+")"});S.append("rect").attr("width",n).attr("height",n).style("fill",t=>x(t.label)).style("stroke",t=>x(t.label)),S.append("text").attr("x",n+p).attr("y",n-p).text(t=>r.getShowData()?`${t.label} [${t.value}]`:t.label);let le=Math.max(...S.selectAll("text").nodes().map(t=>t?.getBoundingClientRect().width??0)),se=d+b+n+p+le,W=ne.node()?.getBoundingClientRect().width??0,ce=d/2-W/2,de=d/2+W/2,z=Math.min(0,ce),F=Math.max(se,de)-z;m.attr("viewBox",`${z} 0 ${F} ${s}`),O(m,s,F,A.useMaxWidth)},"draw"),be={draw:Ae},Ge={parser:De,db:te,renderer:be,styles:$e};export{Ge as diagram};
