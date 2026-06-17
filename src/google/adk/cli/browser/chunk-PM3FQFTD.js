import{a as K}from"./chunk-DMWOYWYQ.js";import{a as Q}from"./chunk-T3Q3QCCV.js";import"./chunk-NQKWI5EB.js";import"./chunk-WR6HISGZ.js";import"./chunk-I4UDKKN5.js";import"./chunk-2DLZXFEQ.js";import"./chunk-JNY2YWG7.js";import"./chunk-XULIXUQL.js";import"./chunk-3NJNOY56.js";import"./chunk-NALL4A3P.js";import{a as q}from"./chunk-TPDTRWWV.js";import{k as H,l as J}from"./chunk-WBLSVR3V.js";import"./chunk-GP6TCC26.js";import{N as M,R as P,S as R,T as I,U as L,V as N,W as B,X as U,Y as V,r as W}from"./chunk-QFMJV7VH.js";import{K as C,N as j,g as o,i as g,r as Z}from"./chunk-JRNAXTJ7.js";import"./chunk-ASJUXEUE.js";import"./chunk-EGBSMT36.js";import{j as O}from"./chunk-RMXJBC7V.js";var X=W.pie,D={sections:new Map,showData:!1,config:X},f=D.sections,y=D.showData,se=structuredClone(X),ce=o(()=>structuredClone(se),"getConfig"),de=o(()=>{f=new Map,y=D.showData,P()},"clear"),pe=o(({label:e,value:a})=>{if(a<0)throw new Error(`"${e}" has invalid value: ${a}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);f.has(e)||(f.set(e,a),g.debug(`added new section: ${e}, with value: ${a}`))},"addSection"),ge=o(()=>f,"getSections"),fe=o(e=>{y=e},"setShowData"),ue=o(()=>y,"getShowData"),Y={getConfig:ce,clear:de,setDiagramTitle:B,getDiagramTitle:U,setAccTitle:R,getAccTitle:I,setAccDescription:L,getAccDescription:N,addSection:pe,getSections:ge,setShowData:fe,getShowData:ue},me=o((e,a)=>{K(e,a),a.setShowData(e.showData),e.sections.map(a.addSection)},"populateDb"),he={parse:o(e=>O(null,null,function*(){let a=yield Q("pie",e);g.debug(a),me(a,Y)}),"parse")},ve=o(e=>`
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
`,"getStyles"),Se=ve,xe=o(e=>{let a=[...e.values()].reduce((r,l)=>r+l,0),$=[...e.entries()].map(([r,l])=>({label:r,value:l})).filter(r=>r.value/a*100>=1).sort((r,l)=>l.value-r.value);return j().value(r=>r.value)($)},"createPieArcs"),we=o((e,a,$,T)=>{g.debug(`rendering pie chart
`+e);let r=T.db,l=V(),A=J(r.getConfig(),l.pie),b=40,n=18,d=4,s=450,u=s,m=q(a),c=m.append("g");c.attr("transform","translate("+u/2+","+s/2+")");let{themeVariables:i}=l,[E]=H(i.pieOuterStrokeWidth);E??=2;let _=A.textPosition,p=Math.min(u,s)/2-b,ee=C().innerRadius(0).outerRadius(p),te=C().innerRadius(p*_).outerRadius(p*_);c.append("circle").attr("cx",0).attr("cy",0).attr("r",p+E/2).attr("class","pieOuterCircle");let h=r.getSections(),ae=xe(h),re=[i.pie1,i.pie2,i.pie3,i.pie4,i.pie5,i.pie6,i.pie7,i.pie8,i.pie9,i.pie10,i.pie11,i.pie12],v=0;h.forEach(t=>{v+=t});let k=ae.filter(t=>(t.data.value/v*100).toFixed(0)!=="0"),S=Z(re);c.selectAll("mySlices").data(k).enter().append("path").attr("d",ee).attr("fill",t=>S(t.data.label)).attr("class","pieCircle"),c.selectAll("mySlices").data(k).enter().append("text").text(t=>(t.data.value/v*100).toFixed(0)+"%").attr("transform",t=>"translate("+te.centroid(t)+")").style("text-anchor","middle").attr("class","slice"),c.append("text").text(r.getDiagramTitle()).attr("x",0).attr("y",-(s-50)/2).attr("class","pieTitleText");let z=[...h.entries()].map(([t,w])=>({label:t,value:w})),x=c.selectAll(".legend").data(z).enter().append("g").attr("class","legend").attr("transform",(t,w)=>{let G=n+d,oe=G*z.length/2,le=12*n,ne=w*G-oe;return"translate("+le+","+ne+")"});x.append("rect").attr("width",n).attr("height",n).style("fill",t=>S(t.label)).style("stroke",t=>S(t.label)),x.append("text").attr("x",n+d).attr("y",n-d).text(t=>r.getShowData()?`${t.label} [${t.value}]`:t.label);let ie=Math.max(...x.selectAll("text").nodes().map(t=>t?.getBoundingClientRect().width??0)),F=u+b+n+d+ie;m.attr("viewBox",`0 0 ${F} ${s}`),M(m,s,F,A.useMaxWidth)},"draw"),Ce={draw:we},_e={parser:he,db:Y,renderer:Ce,styles:Se};export{_e as diagram};
