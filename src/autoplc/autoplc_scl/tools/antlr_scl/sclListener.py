# Generated from scl.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .sclParser import sclParser
else:
    from sclParser import sclParser

# This class defines a complete listener for a parse tree produced by sclParser.
class sclListener(ParseTreeListener):

    # Enter a parse tree produced by sclParser#r.
    def enterR(self, ctx:sclParser.RContext):
        pass

    # Exit a parse tree produced by sclParser#r.
    def exitR(self, ctx:sclParser.RContext):
        pass


    # Enter a parse tree produced by sclParser#whileStatement.
    def enterWhileStatement(self, ctx:sclParser.WhileStatementContext):
        pass

    # Exit a parse tree produced by sclParser#whileStatement.
    def exitWhileStatement(self, ctx:sclParser.WhileStatementContext):
        pass


    # Enter a parse tree produced by sclParser#forStatement.
    def enterForStatement(self, ctx:sclParser.ForStatementContext):
        pass

    # Exit a parse tree produced by sclParser#forStatement.
    def exitForStatement(self, ctx:sclParser.ForStatementContext):
        pass


    # Enter a parse tree produced by sclParser#ifStatement.
    def enterIfStatement(self, ctx:sclParser.IfStatementContext):
        pass

    # Exit a parse tree produced by sclParser#ifStatement.
    def exitIfStatement(self, ctx:sclParser.IfStatementContext):
        pass


    # Enter a parse tree produced by sclParser#whileCondition.
    def enterWhileCondition(self, ctx:sclParser.WhileConditionContext):
        pass

    # Exit a parse tree produced by sclParser#whileCondition.
    def exitWhileCondition(self, ctx:sclParser.WhileConditionContext):
        pass


    # Enter a parse tree produced by sclParser#whileBlockStatements.
    def enterWhileBlockStatements(self, ctx:sclParser.WhileBlockStatementsContext):
        pass

    # Exit a parse tree produced by sclParser#whileBlockStatements.
    def exitWhileBlockStatements(self, ctx:sclParser.WhileBlockStatementsContext):
        pass


    # Enter a parse tree produced by sclParser#repeatStatement.
    def enterRepeatStatement(self, ctx:sclParser.RepeatStatementContext):
        pass

    # Exit a parse tree produced by sclParser#repeatStatement.
    def exitRepeatStatement(self, ctx:sclParser.RepeatStatementContext):
        pass


    # Enter a parse tree produced by sclParser#switchStatement.
    def enterSwitchStatement(self, ctx:sclParser.SwitchStatementContext):
        pass

    # Exit a parse tree produced by sclParser#switchStatement.
    def exitSwitchStatement(self, ctx:sclParser.SwitchStatementContext):
        pass


    # Enter a parse tree produced by sclParser#block.
    def enterBlock(self, ctx:sclParser.BlockContext):
        pass

    # Exit a parse tree produced by sclParser#block.
    def exitBlock(self, ctx:sclParser.BlockContext):
        pass


    # Enter a parse tree produced by sclParser#dataBlock.
    def enterDataBlock(self, ctx:sclParser.DataBlockContext):
        pass

    # Exit a parse tree produced by sclParser#dataBlock.
    def exitDataBlock(self, ctx:sclParser.DataBlockContext):
        pass


    # Enter a parse tree produced by sclParser#typeBlock.
    def enterTypeBlock(self, ctx:sclParser.TypeBlockContext):
        pass

    # Exit a parse tree produced by sclParser#typeBlock.
    def exitTypeBlock(self, ctx:sclParser.TypeBlockContext):
        pass


    # Enter a parse tree produced by sclParser#fbBlock.
    def enterFbBlock(self, ctx:sclParser.FbBlockContext):
        pass

    # Exit a parse tree produced by sclParser#fbBlock.
    def exitFbBlock(self, ctx:sclParser.FbBlockContext):
        pass


    # Enter a parse tree produced by sclParser#fcBlock.
    def enterFcBlock(self, ctx:sclParser.FcBlockContext):
        pass

    # Exit a parse tree produced by sclParser#fcBlock.
    def exitFcBlock(self, ctx:sclParser.FcBlockContext):
        pass


    # Enter a parse tree produced by sclParser#blockName.
    def enterBlockName(self, ctx:sclParser.BlockNameContext):
        pass

    # Exit a parse tree produced by sclParser#blockName.
    def exitBlockName(self, ctx:sclParser.BlockNameContext):
        pass


    # Enter a parse tree produced by sclParser#dbInstanceName.
    def enterDbInstanceName(self, ctx:sclParser.DbInstanceNameContext):
        pass

    # Exit a parse tree produced by sclParser#dbInstanceName.
    def exitDbInstanceName(self, ctx:sclParser.DbInstanceNameContext):
        pass


    # Enter a parse tree produced by sclParser#structType.
    def enterStructType(self, ctx:sclParser.StructTypeContext):
        pass

    # Exit a parse tree produced by sclParser#structType.
    def exitStructType(self, ctx:sclParser.StructTypeContext):
        pass


    # Enter a parse tree produced by sclParser#variableDefinitions.
    def enterVariableDefinitions(self, ctx:sclParser.VariableDefinitionsContext):
        pass

    # Exit a parse tree produced by sclParser#variableDefinitions.
    def exitVariableDefinitions(self, ctx:sclParser.VariableDefinitionsContext):
        pass


    # Enter a parse tree produced by sclParser#variableDefinition.
    def enterVariableDefinition(self, ctx:sclParser.VariableDefinitionContext):
        pass

    # Exit a parse tree produced by sclParser#variableDefinition.
    def exitVariableDefinition(self, ctx:sclParser.VariableDefinitionContext):
        pass


    # Enter a parse tree produced by sclParser#weirdThing.
    def enterWeirdThing(self, ctx:sclParser.WeirdThingContext):
        pass

    # Exit a parse tree produced by sclParser#weirdThing.
    def exitWeirdThing(self, ctx:sclParser.WeirdThingContext):
        pass


    # Enter a parse tree produced by sclParser#constant_assign.
    def enterConstant_assign(self, ctx:sclParser.Constant_assignContext):
        pass

    # Exit a parse tree produced by sclParser#constant_assign.
    def exitConstant_assign(self, ctx:sclParser.Constant_assignContext):
        pass


    # Enter a parse tree produced by sclParser#variableType.
    def enterVariableType(self, ctx:sclParser.VariableTypeContext):
        pass

    # Exit a parse tree produced by sclParser#variableType.
    def exitVariableType(self, ctx:sclParser.VariableTypeContext):
        pass


    # Enter a parse tree produced by sclParser#udtType.
    def enterUdtType(self, ctx:sclParser.UdtTypeContext):
        pass

    # Exit a parse tree produced by sclParser#udtType.
    def exitUdtType(self, ctx:sclParser.UdtTypeContext):
        pass


    # Enter a parse tree produced by sclParser#elementaryType.
    def enterElementaryType(self, ctx:sclParser.ElementaryTypeContext):
        pass

    # Exit a parse tree produced by sclParser#elementaryType.
    def exitElementaryType(self, ctx:sclParser.ElementaryTypeContext):
        pass


    # Enter a parse tree produced by sclParser#arrayType.
    def enterArrayType(self, ctx:sclParser.ArrayTypeContext):
        pass

    # Exit a parse tree produced by sclParser#arrayType.
    def exitArrayType(self, ctx:sclParser.ArrayTypeContext):
        pass


    # Enter a parse tree produced by sclParser#arrayRange.
    def enterArrayRange(self, ctx:sclParser.ArrayRangeContext):
        pass

    # Exit a parse tree produced by sclParser#arrayRange.
    def exitArrayRange(self, ctx:sclParser.ArrayRangeContext):
        pass


    # Enter a parse tree produced by sclParser#dbAssignments.
    def enterDbAssignments(self, ctx:sclParser.DbAssignmentsContext):
        pass

    # Exit a parse tree produced by sclParser#dbAssignments.
    def exitDbAssignments(self, ctx:sclParser.DbAssignmentsContext):
        pass


    # Enter a parse tree produced by sclParser#dbAssignment.
    def enterDbAssignment(self, ctx:sclParser.DbAssignmentContext):
        pass

    # Exit a parse tree produced by sclParser#dbAssignment.
    def exitDbAssignment(self, ctx:sclParser.DbAssignmentContext):
        pass


    # Enter a parse tree produced by sclParser#nonRetainBlock.
    def enterNonRetainBlock(self, ctx:sclParser.NonRetainBlockContext):
        pass

    # Exit a parse tree produced by sclParser#nonRetainBlock.
    def exitNonRetainBlock(self, ctx:sclParser.NonRetainBlockContext):
        pass


    # Enter a parse tree produced by sclParser#blockVarDeclarations.
    def enterBlockVarDeclarations(self, ctx:sclParser.BlockVarDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockVarDeclarations.
    def exitBlockVarDeclarations(self, ctx:sclParser.BlockVarDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockTempVars.
    def enterBlockTempVars(self, ctx:sclParser.BlockTempVarsContext):
        pass

    # Exit a parse tree produced by sclParser#blockTempVars.
    def exitBlockTempVars(self, ctx:sclParser.BlockTempVarsContext):
        pass


    # Enter a parse tree produced by sclParser#blockInOutDeclarations.
    def enterBlockInOutDeclarations(self, ctx:sclParser.BlockInOutDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockInOutDeclarations.
    def exitBlockInOutDeclarations(self, ctx:sclParser.BlockInOutDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockInputDeclarations.
    def enterBlockInputDeclarations(self, ctx:sclParser.BlockInputDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockInputDeclarations.
    def exitBlockInputDeclarations(self, ctx:sclParser.BlockInputDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockOutputDeclarations.
    def enterBlockOutputDeclarations(self, ctx:sclParser.BlockOutputDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockOutputDeclarations.
    def exitBlockOutputDeclarations(self, ctx:sclParser.BlockOutputDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockConstDeclarations.
    def enterBlockConstDeclarations(self, ctx:sclParser.BlockConstDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockConstDeclarations.
    def exitBlockConstDeclarations(self, ctx:sclParser.BlockConstDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockStaticDeclarations.
    def enterBlockStaticDeclarations(self, ctx:sclParser.BlockStaticDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockStaticDeclarations.
    def exitBlockStaticDeclarations(self, ctx:sclParser.BlockStaticDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockConstantDeclarations.
    def enterBlockConstantDeclarations(self, ctx:sclParser.BlockConstantDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockConstantDeclarations.
    def exitBlockConstantDeclarations(self, ctx:sclParser.BlockConstantDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockLabelDeclarations.
    def enterBlockLabelDeclarations(self, ctx:sclParser.BlockLabelDeclarationsContext):
        pass

    # Exit a parse tree produced by sclParser#blockLabelDeclarations.
    def exitBlockLabelDeclarations(self, ctx:sclParser.BlockLabelDeclarationsContext):
        pass


    # Enter a parse tree produced by sclParser#blockFunctionality.
    def enterBlockFunctionality(self, ctx:sclParser.BlockFunctionalityContext):
        pass

    # Exit a parse tree produced by sclParser#blockFunctionality.
    def exitBlockFunctionality(self, ctx:sclParser.BlockFunctionalityContext):
        pass


    # Enter a parse tree produced by sclParser#function_type.
    def enterFunction_type(self, ctx:sclParser.Function_typeContext):
        pass

    # Exit a parse tree produced by sclParser#function_type.
    def exitFunction_type(self, ctx:sclParser.Function_typeContext):
        pass


    # Enter a parse tree produced by sclParser#stat.
    def enterStat(self, ctx:sclParser.StatContext):
        pass

    # Exit a parse tree produced by sclParser#stat.
    def exitStat(self, ctx:sclParser.StatContext):
        pass


    # Enter a parse tree produced by sclParser#functionStatement.
    def enterFunctionStatement(self, ctx:sclParser.FunctionStatementContext):
        pass

    # Exit a parse tree produced by sclParser#functionStatement.
    def exitFunctionStatement(self, ctx:sclParser.FunctionStatementContext):
        pass


    # Enter a parse tree produced by sclParser#labelStatment.
    def enterLabelStatment(self, ctx:sclParser.LabelStatmentContext):
        pass

    # Exit a parse tree produced by sclParser#labelStatment.
    def exitLabelStatment(self, ctx:sclParser.LabelStatmentContext):
        pass


    # Enter a parse tree produced by sclParser#forInitialCondition.
    def enterForInitialCondition(self, ctx:sclParser.ForInitialConditionContext):
        pass

    # Exit a parse tree produced by sclParser#forInitialCondition.
    def exitForInitialCondition(self, ctx:sclParser.ForInitialConditionContext):
        pass


    # Enter a parse tree produced by sclParser#forEndCondition.
    def enterForEndCondition(self, ctx:sclParser.ForEndConditionContext):
        pass

    # Exit a parse tree produced by sclParser#forEndCondition.
    def exitForEndCondition(self, ctx:sclParser.ForEndConditionContext):
        pass


    # Enter a parse tree produced by sclParser#forStepCondition.
    def enterForStepCondition(self, ctx:sclParser.ForStepConditionContext):
        pass

    # Exit a parse tree produced by sclParser#forStepCondition.
    def exitForStepCondition(self, ctx:sclParser.ForStepConditionContext):
        pass


    # Enter a parse tree produced by sclParser#forBlockStatements.
    def enterForBlockStatements(self, ctx:sclParser.ForBlockStatementsContext):
        pass

    # Exit a parse tree produced by sclParser#forBlockStatements.
    def exitForBlockStatements(self, ctx:sclParser.ForBlockStatementsContext):
        pass


    # Enter a parse tree produced by sclParser#ifCondition.
    def enterIfCondition(self, ctx:sclParser.IfConditionContext):
        pass

    # Exit a parse tree produced by sclParser#ifCondition.
    def exitIfCondition(self, ctx:sclParser.IfConditionContext):
        pass


    # Enter a parse tree produced by sclParser#ifBlockStatments.
    def enterIfBlockStatments(self, ctx:sclParser.IfBlockStatmentsContext):
        pass

    # Exit a parse tree produced by sclParser#ifBlockStatments.
    def exitIfBlockStatments(self, ctx:sclParser.IfBlockStatmentsContext):
        pass


    # Enter a parse tree produced by sclParser#switchBlock.
    def enterSwitchBlock(self, ctx:sclParser.SwitchBlockContext):
        pass

    # Exit a parse tree produced by sclParser#switchBlock.
    def exitSwitchBlock(self, ctx:sclParser.SwitchBlockContext):
        pass


    # Enter a parse tree produced by sclParser#switchLabel.
    def enterSwitchLabel(self, ctx:sclParser.SwitchLabelContext):
        pass

    # Exit a parse tree produced by sclParser#switchLabel.
    def exitSwitchLabel(self, ctx:sclParser.SwitchLabelContext):
        pass


    # Enter a parse tree produced by sclParser#switchLabelConstant.
    def enterSwitchLabelConstant(self, ctx:sclParser.SwitchLabelConstantContext):
        pass

    # Exit a parse tree produced by sclParser#switchLabelConstant.
    def exitSwitchLabelConstant(self, ctx:sclParser.SwitchLabelConstantContext):
        pass


    # Enter a parse tree produced by sclParser#switchBlockStatements.
    def enterSwitchBlockStatements(self, ctx:sclParser.SwitchBlockStatementsContext):
        pass

    # Exit a parse tree produced by sclParser#switchBlockStatements.
    def exitSwitchBlockStatements(self, ctx:sclParser.SwitchBlockStatementsContext):
        pass


    # Enter a parse tree produced by sclParser#defualtswitchBlockStatements.
    def enterDefualtswitchBlockStatements(self, ctx:sclParser.DefualtswitchBlockStatementsContext):
        pass

    # Exit a parse tree produced by sclParser#defualtswitchBlockStatements.
    def exitDefualtswitchBlockStatements(self, ctx:sclParser.DefualtswitchBlockStatementsContext):
        pass


    # Enter a parse tree produced by sclParser#assignmentStatement.
    def enterAssignmentStatement(self, ctx:sclParser.AssignmentStatementContext):
        pass

    # Exit a parse tree produced by sclParser#assignmentStatement.
    def exitAssignmentStatement(self, ctx:sclParser.AssignmentStatementContext):
        pass


    # Enter a parse tree produced by sclParser#leftHandAssignment.
    def enterLeftHandAssignment(self, ctx:sclParser.LeftHandAssignmentContext):
        pass

    # Exit a parse tree produced by sclParser#leftHandAssignment.
    def exitLeftHandAssignment(self, ctx:sclParser.LeftHandAssignmentContext):
        pass


    # Enter a parse tree produced by sclParser#rightHandAssignment.
    def enterRightHandAssignment(self, ctx:sclParser.RightHandAssignmentContext):
        pass

    # Exit a parse tree produced by sclParser#rightHandAssignment.
    def exitRightHandAssignment(self, ctx:sclParser.RightHandAssignmentContext):
        pass


    # Enter a parse tree produced by sclParser#expr.
    def enterExpr(self, ctx:sclParser.ExprContext):
        pass

    # Exit a parse tree produced by sclParser#expr.
    def exitExpr(self, ctx:sclParser.ExprContext):
        pass


    # Enter a parse tree produced by sclParser#varName.
    def enterVarName(self, ctx:sclParser.VarNameContext):
        pass

    # Exit a parse tree produced by sclParser#varName.
    def exitVarName(self, ctx:sclParser.VarNameContext):
        pass


    # Enter a parse tree produced by sclParser#functionParameterExpression.
    def enterFunctionParameterExpression(self, ctx:sclParser.FunctionParameterExpressionContext):
        pass

    # Exit a parse tree produced by sclParser#functionParameterExpression.
    def exitFunctionParameterExpression(self, ctx:sclParser.FunctionParameterExpressionContext):
        pass


    # Enter a parse tree produced by sclParser#inputParameter.
    def enterInputParameter(self, ctx:sclParser.InputParameterContext):
        pass

    # Exit a parse tree produced by sclParser#inputParameter.
    def exitInputParameter(self, ctx:sclParser.InputParameterContext):
        pass


    # Enter a parse tree produced by sclParser#outputParameter.
    def enterOutputParameter(self, ctx:sclParser.OutputParameterContext):
        pass

    # Exit a parse tree produced by sclParser#outputParameter.
    def exitOutputParameter(self, ctx:sclParser.OutputParameterContext):
        pass


    # Enter a parse tree produced by sclParser#numberConstant.
    def enterNumberConstant(self, ctx:sclParser.NumberConstantContext):
        pass

    # Exit a parse tree produced by sclParser#numberConstant.
    def exitNumberConstant(self, ctx:sclParser.NumberConstantContext):
        pass


    # Enter a parse tree produced by sclParser#constant.
    def enterConstant(self, ctx:sclParser.ConstantContext):
        pass

    # Exit a parse tree produced by sclParser#constant.
    def exitConstant(self, ctx:sclParser.ConstantContext):
        pass


    # Enter a parse tree produced by sclParser#expressionName.
    def enterExpressionName(self, ctx:sclParser.ExpressionNameContext):
        pass

    # Exit a parse tree produced by sclParser#expressionName.
    def exitExpressionName(self, ctx:sclParser.ExpressionNameContext):
        pass


    # Enter a parse tree produced by sclParser#ambiguousName.
    def enterAmbiguousName(self, ctx:sclParser.AmbiguousNameContext):
        pass

    # Exit a parse tree produced by sclParser#ambiguousName.
    def exitAmbiguousName(self, ctx:sclParser.AmbiguousNameContext):
        pass


    # Enter a parse tree produced by sclParser#ambiguousNext.
    def enterAmbiguousNext(self, ctx:sclParser.AmbiguousNextContext):
        pass

    # Exit a parse tree produced by sclParser#ambiguousNext.
    def exitAmbiguousNext(self, ctx:sclParser.AmbiguousNextContext):
        pass


    # Enter a parse tree produced by sclParser#identifierIndex.
    def enterIdentifierIndex(self, ctx:sclParser.IdentifierIndexContext):
        pass

    # Exit a parse tree produced by sclParser#identifierIndex.
    def exitIdentifierIndex(self, ctx:sclParser.IdentifierIndexContext):
        pass


    # Enter a parse tree produced by sclParser#identifierIndexValue.
    def enterIdentifierIndexValue(self, ctx:sclParser.IdentifierIndexValueContext):
        pass

    # Exit a parse tree produced by sclParser#identifierIndexValue.
    def exitIdentifierIndexValue(self, ctx:sclParser.IdentifierIndexValueContext):
        pass



del sclParser